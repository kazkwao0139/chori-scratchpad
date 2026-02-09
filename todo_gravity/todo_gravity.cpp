/*
 * Todo Gravity — C++ Implementation
 * 무거운 일이 먼저 온다.
 *
 * 키스트로크 타이밍에서 중요도를 추정한다.
 * |z-score| = 평소 대비 얼마나 다르게 타이핑했는가.
 * 방향 무관 — 급해서 빨라져도, 고민해서 느려져도, 평소와 다르면 중요하다.
 *
 * 근거: Epp et al. (CHI 2011) — 키스트로크 리듬으로 감정 상태 77-88% 분류
 *       PLOS ONE (2015) — arousal이 타이핑 타이밍에 유의미한 영향
 *       Freihaut (2021, N=924) — 개인별 베이스라인 필수
 *
 * Build: g++ -std=c++17 -O2 -o todo_gravity todo_gravity.cpp
 * (Windows): cl /std:c++17 /EHsc /utf-8 todo_gravity.cpp
 */

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <chrono>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <optional>

#ifdef _WIN32
#include <conio.h>
#include <windows.h>
#else
#include <termios.h>
#include <unistd.h>
#endif

// ─── Platform Input ──────────────────────────────────

#ifdef _WIN32
void setupConsole() {
    SetConsoleOutputCP(65001);
    SetConsoleCP(65001);
    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    DWORD mode;
    GetConsoleMode(hOut, &mode);
    SetConsoleMode(hOut, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
}

int getChar() { return _getch(); }
bool keyReady() { return _kbhit() != 0; }
#else
static struct termios oldTerm, newTerm;
void setupConsole() {
    tcgetattr(STDIN_FILENO, &oldTerm);
    newTerm = oldTerm;
    newTerm.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &newTerm);
}
void restoreConsole() { tcsetattr(STDIN_FILENO, TCSANOW, &oldTerm); }
int getChar() { return getchar(); }
bool keyReady() {
    fd_set fds;
    FD_ZERO(&fds);
    FD_SET(STDIN_FILENO, &fds);
    struct timeval tv = {0, 0};
    return select(1, &fds, NULL, NULL, &tv) > 0;
}
#endif

using Clock = std::chrono::high_resolution_clock;
using Ms = std::chrono::duration<double, std::milli>;

// ─── Stats ───────────────────────────────────────────

struct Stats {
    double mean = 0;
    double stddev = 1;
    int count = 0;

    void update(double value) {
        count++;
        double delta = value - mean;
        mean += delta / count;
        if (count > 1) {
            // Welford's online algorithm
            double delta2 = value - mean;
            double m2 = stddev * stddev * (count - 2) + delta * delta2;
            stddev = std::sqrt(m2 / (count - 1));
        }
    }

    double zScore(double value) const {
        if (count < 2 || stddev < 0.001) return 0;
        return std::abs(value - mean) / stddev;
    }
};

// ─── Keystroke Features ──────────────────────────────

struct EntryFeatures {
    double meanInterval = 0;  // mean inter-key interval (ms)
    double speed = 0;         // chars per second
    double bsRatio = 0;       // backspace ratio
    double totalTime = 0;     // total composition time (ms)
};

// ─── Todo Item ───────────────────────────────────────

struct Todo {
    int id;
    std::string text;
    EntryFeatures features;
    double importance;
    bool hasImportance;

    std::string serialize() const {
        std::ostringstream oss;
        oss << id << "|" << text << "|"
            << features.meanInterval << "|"
            << features.speed << "|"
            << features.bsRatio << "|"
            << features.totalTime << "|"
            << importance << "|"
            << (hasImportance ? 1 : 0);
        return oss.str();
    }

    static std::optional<Todo> deserialize(const std::string& line) {
        std::istringstream iss(line);
        Todo t;
        std::string field;
        int hi;
        try {
            std::getline(iss, field, '|'); t.id = std::stoi(field);
            std::getline(iss, t.text, '|');
            std::getline(iss, field, '|'); t.features.meanInterval = std::stod(field);
            std::getline(iss, field, '|'); t.features.speed = std::stod(field);
            std::getline(iss, field, '|'); t.features.bsRatio = std::stod(field);
            std::getline(iss, field, '|'); t.features.totalTime = std::stod(field);
            std::getline(iss, field, '|'); t.importance = std::stod(field);
            std::getline(iss, field, '|'); hi = std::stoi(field);
            t.hasImportance = (hi == 1);
            return t;
        } catch (...) {
            return std::nullopt;
        }
    }
};

// ─── App ─────────────────────────────────────────────

class TodoGravity {
    std::vector<Todo> todos;
    Stats intervalStats;
    Stats speedStats;
    Stats bsStats;
    Stats timeStats;
    int nextId = 1;
    int entryCount = 0;
    static constexpr int BASELINE_N = 5;
    static constexpr const char* SAVE_FILE = "todo_gravity_data.txt";

    // Weights from paper findings: hold/flight most predictive
    static constexpr double W_INTERVAL = 0.35;
    static constexpr double W_SPEED    = 0.30;
    static constexpr double W_BS       = 0.15;
    static constexpr double W_TIME     = 0.20;

public:
    TodoGravity() { load(); }

    // ─── Keystroke Capture ────────────────────────────

    struct InputResult {
        std::string text;
        EntryFeatures features;
    };

    InputResult captureInput() {
        std::string text;
        std::vector<double> intervals;
        int backspaces = 0;
        int chars = 0;

        auto start = Clock::now();
        auto lastKey = start;
        bool first = true;

        std::cout << "  > \033[90m";

        while (true) {
            int ch = getChar();

            if (ch == '\r' || ch == '\n') {
                break;
            }

            auto now = Clock::now();

            if (ch == 8 || ch == 127) { // backspace
                if (!text.empty()) {
                    text.pop_back();
                    backspaces++;
                    std::cout << "\b \b";
                }
                lastKey = now;
                continue;
            }

            if (ch < 32) continue; // ignore control chars

            if (!first) {
                double interval = Ms(now - lastKey).count();
                if (interval < 2000) {
                    intervals.push_back(interval);
                }
            }
            first = false;
            lastKey = now;
            chars++;

            text += static_cast<char>(ch);
            std::cout << static_cast<char>(ch);
        }

        std::cout << "\033[0m" << std::endl;

        auto end = Clock::now();
        double totalTime = Ms(end - start).count();

        EntryFeatures f;
        if (!intervals.empty()) {
            f.meanInterval = std::accumulate(intervals.begin(), intervals.end(), 0.0) / intervals.size();
        }
        f.speed = chars > 0 ? (chars / (totalTime / 1000.0)) : 0;
        f.bsRatio = chars > 0 ? static_cast<double>(backspaces) / chars : 0;
        f.totalTime = totalTime;

        return {text, f};
    }

    // ─── Importance Scoring ──────────────────────────

    double computeImportance(const EntryFeatures& f) {
        if (entryCount < BASELINE_N) return -1; // still collecting

        double z = 0;
        double wSum = 0;

        auto addZ = [&](double val, const Stats& s, double w) {
            double zv = s.zScore(val);
            z += zv * w;
            wSum += w;
        };

        addZ(f.meanInterval, intervalStats, W_INTERVAL);
        addZ(f.speed, speedStats, W_SPEED);
        addZ(f.bsRatio, bsStats, W_BS);
        addZ(f.totalTime, timeStats, W_TIME);

        return wSum > 0 ? z / wSum : 0;
    }

    void updateBaseline(const EntryFeatures& f) {
        intervalStats.update(f.meanInterval);
        speedStats.update(f.speed);
        bsStats.update(f.bsRatio);
        timeStats.update(f.totalTime);
        entryCount++;
    }

    // ─── Actions ─────────────────────────────────────

    void add() {
        auto [text, features] = captureInput();
        if (text.empty()) return;

        double imp = computeImportance(features);
        updateBaseline(features);

        Todo todo{nextId++, text, features, imp, imp >= 0};

        // Recalculate when baseline just completed
        if (entryCount == BASELINE_N) {
            for (auto& t : todos) {
                t.importance = computeImportance(t.features);
                t.hasImportance = (t.importance >= 0);
            }
        }

        todos.push_back(todo);
        sortTodos();
        save();

        std::cout << "  + " << text;
        if (todo.hasImportance) {
            std::cout << " \033[90m(mass: " << std::fixed << std::setprecision(2)
                      << todo.importance << ")\033[0m";
        }
        std::cout << std::endl;
    }

    void complete(int idx) {
        if (idx < 0 || idx >= static_cast<int>(todos.size())) {
            std::cout << "  invalid index" << std::endl;
            return;
        }
        std::cout << "  v " << todos[idx].text << std::endl;
        todos.erase(todos.begin() + idx);
        save();
    }

    void remove(int idx) {
        if (idx < 0 || idx >= static_cast<int>(todos.size())) {
            std::cout << "  invalid index" << std::endl;
            return;
        }
        std::cout << "  x " << todos[idx].text << std::endl;
        todos.erase(todos.begin() + idx);
        save();
    }

    // ─── Display ─────────────────────────────────────

    void display() {
        std::cout << std::endl;

        if (entryCount < BASELINE_N) {
            std::cout << "  \033[90mbaseline: " << entryCount << "/" << BASELINE_N
                      << "\033[0m" << std::endl;
        }

        if (todos.empty()) {
            std::cout << "  \033[90mno todos\033[0m" << std::endl << std::endl;
            return;
        }

        double maxImp = 0.01;
        for (const auto& t : todos) {
            if (t.hasImportance && t.importance > maxImp) maxImp = t.importance;
        }

        std::cout << "\033[90m  " << std::string(48, '-') << "\033[0m" << std::endl;

        for (int i = 0; i < static_cast<int>(todos.size()); i++) {
            const auto& t = todos[i];

            // Mass bar
            int barLen = 0;
            std::string impStr = " ... ";
            if (t.hasImportance) {
                double norm = std::min(t.importance / maxImp, 1.0);
                barLen = static_cast<int>(norm * 12);
                std::ostringstream oss;
                oss << std::fixed << std::setprecision(2) << t.importance;
                impStr = oss.str();
            }

            std::string bar(barLen, '#');
            std::string space(12 - barLen, ' ');

            std::cout << "  " << std::setw(2) << i << "  "
                      << "\033[90m" << bar << space << "\033[0m "
                      << std::setw(5) << impStr << "  "
                      << t.text << std::endl;
        }

        std::cout << "\033[90m  " << std::string(48, '-') << "\033[0m" << std::endl;
        std::cout << std::endl;
    }

    void showHelp() {
        std::cout << "  \033[90m[a]dd  [c]# complete  [d]# delete  [q]uit\033[0m" << std::endl;
    }

    // ─── Sort ────────────────────────────────────────

    void sortTodos() {
        std::stable_sort(todos.begin(), todos.end(), [](const Todo& a, const Todo& b) {
            double ia = a.hasImportance ? a.importance : -1;
            double ib = b.hasImportance ? b.importance : -1;
            return ib < ia;
        });
    }

    // ─── Persistence ─────────────────────────────────

    void save() {
        std::ofstream f(SAVE_FILE);
        if (!f) return;

        // Save stats
        f << "STATS|" << entryCount << "|"
          << intervalStats.mean << "|" << intervalStats.stddev << "|" << intervalStats.count << "|"
          << speedStats.mean << "|" << speedStats.stddev << "|" << speedStats.count << "|"
          << bsStats.mean << "|" << bsStats.stddev << "|" << bsStats.count << "|"
          << timeStats.mean << "|" << timeStats.stddev << "|" << timeStats.count << "|"
          << nextId << "\n";

        for (const auto& t : todos) {
            f << "TODO|" << t.serialize() << "\n";
        }
    }

    void load() {
        std::ifstream f(SAVE_FILE);
        if (!f) return;

        std::string line;
        while (std::getline(f, line)) {
            if (line.substr(0, 6) == "STATS|") {
                std::istringstream iss(line.substr(6));
                std::string field;
                try {
                    std::getline(iss, field, '|'); entryCount = std::stoi(field);
                    std::getline(iss, field, '|'); intervalStats.mean = std::stod(field);
                    std::getline(iss, field, '|'); intervalStats.stddev = std::stod(field);
                    std::getline(iss, field, '|'); intervalStats.count = std::stoi(field);
                    std::getline(iss, field, '|'); speedStats.mean = std::stod(field);
                    std::getline(iss, field, '|'); speedStats.stddev = std::stod(field);
                    std::getline(iss, field, '|'); speedStats.count = std::stoi(field);
                    std::getline(iss, field, '|'); bsStats.mean = std::stod(field);
                    std::getline(iss, field, '|'); bsStats.stddev = std::stod(field);
                    std::getline(iss, field, '|'); bsStats.count = std::stoi(field);
                    std::getline(iss, field, '|'); timeStats.mean = std::stod(field);
                    std::getline(iss, field, '|'); timeStats.stddev = std::stod(field);
                    std::getline(iss, field, '|'); timeStats.count = std::stoi(field);
                    std::getline(iss, field, '|'); nextId = std::stoi(field);
                } catch (...) {}
            }
            else if (line.substr(0, 5) == "TODO|") {
                auto todo = Todo::deserialize(line.substr(5));
                if (todo) todos.push_back(*todo);
            }
        }
        sortTodos();
    }

    // ─── Main Loop ───────────────────────────────────

    void run() {
        std::cout << std::endl;
        std::cout << "  Todo Gravity" << std::endl;
        std::cout << "  \033[90mheavy things come first\033[0m" << std::endl;

        while (true) {
            display();
            showHelp();
            std::cout << "  ";

            std::string cmd;
            std::getline(std::cin, cmd);

            if (cmd.empty()) continue;

            char action = cmd[0];
            std::string arg = cmd.size() > 1 ? cmd.substr(1) : "";

            // Trim
            while (!arg.empty() && arg[0] == ' ') arg.erase(0, 1);

            switch (action) {
                case 'q': case 'Q':
                    std::cout << "  bye" << std::endl << std::endl;
                    return;

                case 'a': case 'A':
                    add();
                    break;

                case 'c': case 'C':
                    try { complete(std::stoi(arg)); } catch (...) { std::cout << "  c [#]" << std::endl; }
                    break;

                case 'd': case 'D':
                    try { remove(std::stoi(arg)); } catch (...) { std::cout << "  d [#]" << std::endl; }
                    break;

                default:
                    std::cout << "  ?" << std::endl;
            }
        }
    }
};

int main() {
    setupConsole();
    TodoGravity app;
    app.run();
#ifndef _WIN32
    restoreConsole();
#endif
    return 0;
}
