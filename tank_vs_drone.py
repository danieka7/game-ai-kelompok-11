"""
=========================================================================
 TANK vs DRONE — UCS / A* Pathfinding (versi Python + pygame)
=========================================================================
Port dari versi HTML/JavaScript, dengan MODIFIKASI PERILAKU DRONE:

  1) Drone TIDAK tahu posisi tank sejak awal permainan (berbeda dari versi
     JS asli yang langsung memanggil search(grid, drone, tank, ...) ke
     posisi tank tiap giliran). Sekarang drone hanya tahu posisi tank jika
     tank berada di dalam `vision_radius` DAN tidak ada penghalang pandang
     (pohon/batu/kamp/reruntuhan) di antara keduanya (line-of-sight).

  2) Selama tank belum terdeteksi, drone menjalankan POLA PENCARIAN yang
     sudah DISPESIFIKASIKAN LEBIH DAHULU (predetermined search pattern),
     yaitu pola "boustrophedon" / pola menyisir sawah (baris demi baris,
     arah bolak-balik) — pola nyata yang dipakai dalam misi SAR (search
     and rescue) agar seluruh area tersapu tanpa celah, tanpa perlu tahu
     di mana target berada. Jalur & jejak pola pencarian ini digambar
     dengan WARNA BERBEDA dari jalur kejar (chase path).

  3) Begitu tank terdeteksi, drone berpindah ke mode MENGEJAR (chasing)
     memakai UCS/A* seperti versi asli. TAPI karena tank terus bergerak,
     drone mengulang (re-run) algoritma pencarian jalur SETIAP TICK agar
     rencana rutenya selalu mengikuti posisi tank yang terbaru — drone
     tidak pernah "berhenti mencari" walau sudah tahu posisi tank.

  4) Jika kontak dengan tank hilang (tank keluar dari jangkauan pandang,
     atau pandangan terhalang), drone otomatis kembali menyisir memakai
     pola pencarian preset, MELANJUTKAN dari titik pola yang terakhir
     ia sisir (bukan mengulang dari nol), sehingga cakupan area tetap
     efisien.

Jalankan dengan:  pip install pygame   lalu   python tank_vs_drone.py
=========================================================================
"""

import math
import random
import time
import heapq
from collections import deque

import pygame


# =========================================================================
# BAGIAN 1 — KONSTANTA DUNIA & JENIS MEDAN
# =========================================================================
# Setiap sel grid punya kode medan yang menentukan apakah bisa dilewati
# dan berapa biayanya. Biaya inilah yang menjadi g(n) di UCS maupun A*.

ROWS, COLS, CELL = 16, 22, 30

DIRT      = 0   # tanah terbuka, biaya 1
TREE      = 1   # pohon, terblokir (juga menghalangi pandang)
ROCK      = 2   # bebatuan, terblokir (juga menghalangi pandang)
RIVER     = 3   # sungai, terblokir kecuali di sel jembatan
BRIDGE    = 4   # jembatan, biaya 3
ARTILLERY = 5   # artileri musuh yang ditinggalkan, terblokir
CAMP      = 6   # kamp tentara (blok 2x2), terblokir (menghalangi pandang)
RUIN      = 7   # bangunan runtuh (blok 2x2), terblokir (menghalangi pandang)

TERRAIN_COST = {DIRT: 1, BRIDGE: 3}
BLOCKED_MOVEMENT = {TREE, ROCK, RIVER, ARTILLERY, CAMP, RUIN}
# Objek padat/tinggi yang juga menghalangi PANDANGAN drone (sungai & area
# artileri terbuka tidak menghalangi pandang walau menghalangi gerak).
BLOCKED_SIGHT = {TREE, ROCK, CAMP, RUIN}


def is_blocked(terrain):
    return terrain in BLOCKED_MOVEMENT


# =========================================================================
# BAGIAN 2 — PALET WARNA (setara variabel CSS pada versi HTML)
# =========================================================================
COLORS = {
    "bg_deep":        (27, 26, 21),
    "panel":          (35, 32, 25),
    "panel_border":   (58, 52, 39),
    "text_cream":     (232, 225, 207),
    "text_dim":       (167, 158, 136),
    "accent_signal":  (224, 134, 43),

    "dirt":           (138, 122, 84),
    "dirt_alt":       (127, 111, 75),
    "tree_canopy":    (75, 90, 46),
    "tree_trunk":     (74, 54, 35),
    "rock":           (110, 106, 98),
    "rock_dark":      (84, 81, 74),
    "river":          (74, 102, 112),
    "river_alt":      (82, 112, 129),
    "bridge":         (122, 91, 58),
    "artillery_body": (62, 65, 54),
    "artillery_barrel": (42, 44, 36),
    "camp_tent":      (140, 122, 78),
    "camp_tent_dark": (110, 95, 60),
    "ruin_wall":      (122, 117, 106),
    "ruin_crack":     (78, 74, 66),

    "tank":           (92, 107, 60),
    "tank_dark":      (64, 73, 42),
    "drone":          (193, 68, 58),
    "drone_dark":     (140, 46, 39),

    # Jalur kejar (chase path) — sama seperti "--path" pada versi asli.
    "chase_path":     (240, 212, 139),
    # BARU: jalur & jejak pola pencarian (search pattern) — warna BEDA
    # dari jalur kejar, agar kedua mode terlihat jelas berbeda di layar.
    "search_active":  (110, 168, 254),   # biru terang: segmen jalur aktif
    "search_trail":   (94, 181, 199),    # tosca redup: jejak area tersapu
    "vision_ring":    (224, 134, 43),
    "danger":         (193, 68, 58),
}


# =========================================================================
# BAGIAN 3 — UTILITAS ACAK & PEMBANGKIT MEDAN PERANG
# =========================================================================
# Urutan pembuatan sama seperti versi JS: sungai berkelok + 2 jembatan ->
# pohon acak -> bebatuan acak -> artileri tunggal acak -> 2 kamp (2x2) ->
# 2 bangunan runtuh (2x2) -> validasi keterhubungan lewat BFS, ulangi jika
# titik tank & drone terputus.

def rand_int(lo, hi):
    return random.randint(lo, hi)


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def pick_distinct_rows(count, total):
    chosen = set()
    while len(chosen) < count:
        chosen.add(rand_int(1, total - 2))
    return list(chosen)


def can_place_block_2x2(grid, r, c):
    if r + 1 >= ROWS or c + 1 >= COLS:
        return False
    for dr in range(2):
        for dc in range(2):
            if grid[r + dr][c + dc] != DIRT:
                return False
    return True


def is_reachable(grid, start, goal):
    """BFS sederhana untuk memastikan tank & drone tidak terjebak di area
    yang saling terputus setelah medan dibuat secara acak."""
    visited = [[False] * COLS for _ in range(ROWS)]
    q = deque([start])
    visited[start[0]][start[1]] = True
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while q:
        cur = q.popleft()
        if cur == goal:
            return True
        for dr, dc in dirs:
            nr, nc = cur[0] + dr, cur[1] + dc
            if 0 <= nr < ROWS and 0 <= nc < COLS and not visited[nr][nc]:
                if not is_blocked(grid[nr][nc]):
                    visited[nr][nc] = True
                    q.append((nr, nc))
    return False


def generate_map(start_pos, goal_pos):
    grid = None
    attempt = 0
    while True:
        attempt += 1
        grid = [[DIRT] * COLS for _ in range(ROWS)]

        # 1) sungai berkelok
        river_col = COLS // 2 + rand_int(-2, 2)
        river_cells = []
        for r in range(ROWS):
            river_col = clamp(river_col + rand_int(-1, 1), 2, COLS - 3)
            grid[r][river_col] = RIVER
            river_cells.append((r, river_col))

        # 2) dua jembatan
        for br in pick_distinct_rows(2, ROWS):
            r, c = river_cells[br]
            grid[r][c] = BRIDGE

        # 3) pohon (~8%)
        tree_budget = int(ROWS * COLS * 0.08)
        while tree_budget > 0:
            r, c = rand_int(0, ROWS - 1), rand_int(0, COLS - 1)
            if grid[r][c] == DIRT:
                grid[r][c] = TREE
                tree_budget -= 1

        # 4) bebatuan (~6%)
        rock_budget = int(ROWS * COLS * 0.06)
        while rock_budget > 0:
            r, c = rand_int(0, ROWS - 1), rand_int(0, COLS - 1)
            if grid[r][c] == DIRT:
                grid[r][c] = ROCK
                rock_budget -= 1

        # 5) artileri musuh tak terpakai — 5 unit tersebar satu sel
        artillery_budget = 5
        while artillery_budget > 0:
            r, c = rand_int(0, ROWS - 1), rand_int(0, COLS - 1)
            if grid[r][c] == DIRT:
                grid[r][c] = ARTILLERY
                artillery_budget -= 1

        # 6) kamp tentara — 2 blok 2x2
        camp_budget, safety = 2, 400
        while camp_budget > 0 and safety > 0:
            safety -= 1
            r, c = rand_int(0, ROWS - 2), rand_int(0, COLS - 2)
            if can_place_block_2x2(grid, r, c):
                grid[r][c] = grid[r][c + 1] = grid[r + 1][c] = grid[r + 1][c + 1] = CAMP
                camp_budget -= 1

        # 7) bangunan runtuh — 2 blok 2x2
        ruin_budget, safety = 2, 400
        while ruin_budget > 0 and safety > 0:
            safety -= 1
            r, c = rand_int(0, ROWS - 2), rand_int(0, COLS - 2)
            if can_place_block_2x2(grid, r, c):
                grid[r][c] = grid[r][c + 1] = grid[r + 1][c] = grid[r + 1][c + 1] = RUIN
                ruin_budget -= 1

        # titik awal tank & drone selalu tanah kosong
        grid[start_pos[0]][start_pos[1]] = DIRT
        grid[goal_pos[0]][goal_pos[1]] = DIRT

        if is_reachable(grid, start_pos, goal_pos) or attempt >= 25:
            return grid


# =========================================================================
# BAGIAN 4 — ALGORITMA PENCARIAN: UCS & A* DALAM SATU FUNGSI
# =========================================================================
# UCS adalah kasus khusus A* dengan h(n) = 0 untuk semua node, jadi satu
# fungsi search() melayani kedua algoritma tinggal ganti heuristiknya.
# f(n) = g(n) + h(n)
#
# Catatan implementasi: versi JS memakai linear-scan untuk mencari node
# ber-f terkecil di dalam open list. Di Python kita pakai `heapq` (min-heap)
# yang secara logis setara tapi jauh lebih efisien untuk grid yang besar.
# `counter` dipakai sebagai pemecah seri (tie-breaker) supaya heapq tidak
# perlu membandingkan tuple node ketika nilai f sama persis.

def neighbors(node, grid):
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # gerak 4 arah
    result = []
    for dr, dc in dirs:
        nr, nc = node[0] + dr, node[1] + dc
        if 0 <= nr < ROWS and 0 <= nc < COLS:
            terrain = grid[nr][nc]
            if not is_blocked(terrain):
                result.append(((nr, nc), TERRAIN_COST[terrain]))
    return result


HEURISTICS = {
    # Manhattan = biaya asli persis di area tanpa rintangan pada gerak
    # 4-arah -> paling informatif namun tetap admissible.
    "manhattan": lambda a, b: abs(a[0] - b[0]) + abs(a[1] - b[1]),
    # Euclidean selalu <= Manhattan di grid 4-arah -> admissible tapi longgar.
    "euclidean": lambda a, b: math.hypot(a[0] - b[0], a[1] - b[1]),
    # Chebyshev = max(dr,dc), juga <= Manhattan -> admissible, paling longgar.
    "chebyshev": lambda a, b: max(abs(a[0] - b[0]), abs(a[1] - b[1])),
    # h(n) = 0 untuk semua n -> inilah yang membuat A* berperilaku sebagai UCS.
    "zero": lambda a, b: 0,
}


def search(grid, start, goal, algorithm="astar", heuristic_name="manhattan"):
    t0 = time.perf_counter()
    h_func = HEURISTICS["zero"] if algorithm == "ucs" else HEURISTICS[heuristic_name]

    counter = 0
    open_heap = [(h_func(start, goal), counter, start, 0)]  # (f, tie, node, g)
    came_from = {}
    g_score = {start: 0}
    closed = set()
    expanded_count = 0

    while open_heap:
        f, _, current, g = heapq.heappop(open_heap)

        if current in closed:
            continue
        closed.add(current)
        expanded_count += 1

        if current == goal:
            path = [current]
            walker = current
            while walker in came_from:
                walker = came_from[walker]
                path.append(walker)
            path.reverse()
            return {
                "path": path,
                "cost": g_score[current],
                "expanded": expanded_count,
                "time_ms": (time.perf_counter() - t0) * 1000,
                "found": True,
            }

        for nb, step_cost in neighbors(current, grid):
            if nb in closed:
                continue
            tentative_g = g + step_cost
            if nb not in g_score or tentative_g < g_score[nb]:
                g_score[nb] = tentative_g
                came_from[nb] = current
                counter += 1
                heapq.heappush(open_heap, (tentative_g + h_func(nb, goal), counter, nb, tentative_g))

    return {
        "path": [],
        "cost": float("inf"),
        "expanded": expanded_count,
        "time_ms": (time.perf_counter() - t0) * 1000,
        "found": False,
    }


# =========================================================================
# BAGIAN 5 — MODIFIKASI UTAMA: DETEKSI TERBATAS & POLA PENCARIAN PRESET
# =========================================================================

def bresenham_line(r0, c0, r1, c1):
    """Algoritma garis Bresenham: menghasilkan daftar sel grid yang dilalui
    garis lurus dari (r0,c0) ke (r1,c1). Dipakai untuk uji line-of-sight."""
    points = []
    dr, dc = abs(r1 - r0), abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dr - dc
    r, c = r0, c0
    while True:
        points.append((r, c))
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r += sr
        if e2 < dr:
            err += dr
            c += sc
    return points


def has_line_of_sight(grid, a, b):
    """Drone hanya bisa 'melihat' tank kalau tidak ada penghalang padat
    (pohon/batu/kamp/reruntuhan) di antara posisi drone dan tank. Sungai
    tidak menghalangi pandang walau menghalangi gerak."""
    for (r, c) in bresenham_line(a[0], a[1], b[0], b[1])[1:-1]:
        if grid[r][c] in BLOCKED_SIGHT:
            return False
    return True


def generate_search_pattern(grid):
    """
    MODIFIKASI: membuat daftar titik (waypoint) yang harus dikunjungi
    drone SECARA BERURUTAN mengikuti pola 'boustrophedon' (pola menyisir
    sawah / lawnmower pattern): baris demi baris, arah bolak-balik.

    Ini adalah pola pencarian sistematis yang nyata dipakai dalam misi
    SAR (search and rescue) supaya seluruh area tersapu tanpa celah,
    TANPA perlu tahu di mana target sebenarnya berada. Karena pola ini
    "dispesifikasikan terlebih dahulu" (predetermined), urutannya tetap
    dan drone akan melanjutkan dari titik terakhir jika sempat terganggu
    (misalnya sempat mengejar tank lalu kehilangan kontak).
    """
    waypoints = []
    for r in range(ROWS):
        cols = range(COLS) if r % 2 == 0 else range(COLS - 1, -1, -1)
        for c in cols:
            if not is_blocked(grid[r][c]):
                waypoints.append((r, c))
    return waypoints


class Drone:
    """
    Drone punya dua mode:

      SEARCHING - drone belum tahu di mana tank berada. Ia berjalan
                  mengikuti pola pencarian preset (boustrophedon),
                  memakai UCS/A* hanya untuk NAVIGASI menuju waypoint
                  berikutnya (karena medan penuh rintangan).

      CHASING   - tank sudah terdeteksi (dalam radius pandang & tidak
                  terhalang). Drone mengejar tank langsung memakai
                  UCS/A*. Karena tank terus bergerak, jalur ini DIHITUNG
                  ULANG SETIAP TICK -> drone tidak pernah berhenti
                  "mencari", hanya berganti target pencarian dari
                  waypoint pola menjadi posisi tank saat ini.

    Begitu kontak hilang (tank keluar radius / pandang terhalang), drone
    otomatis kembali ke SEARCHING dan MELANJUTKAN pola dari indeks
    terakhir (bukan mulai dari nol lagi).
    """

    SEARCHING = "SEARCHING"
    CHASING = "CHASING"

    def __init__(self, pos, grid, vision_radius=4):
        self.pos = pos
        self.grid = grid
        self.vision_radius = vision_radius
        self.state = Drone.SEARCHING

        self.search_pattern = generate_search_pattern(grid)
        self.search_index = 0

        self.current_path = []      # jalur yang sedang diikuti saat ini (untuk digambar)
        self.visited_cells = set()  # jejak permanen area yang sudah pernah disisir
        self.last_stats = None

    def rebuild_pattern(self, grid):
        """Dipanggil saat medan baru dibuat (grid berubah total)."""
        self.grid = grid
        self.search_pattern = generate_search_pattern(grid)
        self.search_index = 0
        self.visited_cells.clear()
        self.current_path = []
        self.state = Drone.SEARCHING

    def can_detect(self, tank_pos, grid):
        dist = math.hypot(self.pos[0] - tank_pos[0], self.pos[1] - tank_pos[1])
        if dist > self.vision_radius:
            return False
        return has_line_of_sight(grid, self.pos, tank_pos)

    def take_turn(self, grid, tank_pos, algorithm, heuristic):
        """Satu langkah simulasi drone. Dipanggil berulang kali (setiap
        tick waktu tetap) selama permainan berjalan, TERLEPAS dari apakah
        tank baru saja bergerak atau tidak -> inilah yang membuat drone
        'terus melakukan pencarian' secara berkelanjutan."""
        detected = self.can_detect(tank_pos, grid)

        if detected:
            self.state = Drone.CHASING
            # MODIFIKASI: walau tank sudah "diketahui", algoritma pencarian
            # (UCS/A*) tetap dijalankan ULANG setiap tick, karena posisi
            # tank yang jadi goal selalu berubah (tank terus bergerak).
            result = search(grid, self.pos, tank_pos, algorithm, heuristic)
            self.current_path = result["path"] if result["found"] else []
            if len(self.current_path) > 1:
                self.pos = self.current_path[1]
            self.visited_cells.add(self.pos)
            self.last_stats = result
            return result

        # Tank tidak terdeteksi -> (lanjutkan) menyisir sesuai pola preset.
        self.state = Drone.SEARCHING
        return self._continue_search_pattern(grid, algorithm, heuristic)

    def _continue_search_pattern(self, grid, algorithm, heuristic):
        if not self.search_pattern:
            return {"found": False, "path": [], "expanded": 0, "time_ms": 0.0, "cost": float("inf")}

        target = self.search_pattern[self.search_index]

        # Kalau sudah sampai di waypoint saat ini, maju ke waypoint
        # berikutnya. Indeks dibuat melingkar (modulo) supaya drone terus
        # menyisir berulang selama permainan berjalan, bukan berhenti
        # setelah satu putaran penuh — sesuai permintaan "terus mencari".
        if self.pos == target:
            self.visited_cells.add(self.pos)
            self.search_index = (self.search_index + 1) % len(self.search_pattern)
            target = self.search_pattern[self.search_index]

        result = search(grid, self.pos, target, algorithm, heuristic)
        self.current_path = result["path"] if result["found"] else []
        if len(self.current_path) > 1:
            self.pos = self.current_path[1]
        self.visited_cells.add(self.pos)
        self.last_stats = result
        return result


# =========================================================================
# BAGIAN 6 — GAMBAR VEKTOR BANTU (rotasi manual, setara ctx.rotate di JS)
# =========================================================================

def rotate_point(x, y, theta):
    ct, st = math.cos(theta), math.sin(theta)
    return x * ct - y * st, x * st + y * ct


def rotated_rect_points(cx, cy, w, h, theta):
    local = [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]
    pts = []
    for x, y in local:
        rx, ry = rotate_point(x, y, theta)
        pts.append((cx + rx, cy + ry))
    return pts


def draw_cell(surface, terrain, r, c):
    x, y = c * CELL, r * CELL
    checker = (r + c) % 2 == 0

    base = COLORS["dirt"] if checker else COLORS["dirt_alt"]
    pygame.draw.rect(surface, base, (x, y, CELL, CELL))

    if terrain == TREE:
        pygame.draw.rect(surface, COLORS["tree_trunk"], (x + CELL / 2 - 2, y + CELL * 0.55, 4, CELL * 0.4))
        pygame.draw.circle(surface, COLORS["tree_canopy"], (int(x + CELL / 2), int(y + CELL * 0.42)), int(CELL * 0.36))
    elif terrain == ROCK:
        pygame.draw.ellipse(surface, COLORS["rock"],
                             (x + CELL * 0.12, y + CELL * 0.27, CELL * 0.76, CELL * 0.56))
        pygame.draw.ellipse(surface, COLORS["rock_dark"],
                             (x + CELL * 0.44, y + CELL * 0.48, CELL * 0.32, CELL * 0.24))
    elif terrain == RIVER:
        pygame.draw.rect(surface, COLORS["river"] if checker else COLORS["river_alt"], (x, y, CELL, CELL))
    elif terrain == BRIDGE:
        pygame.draw.rect(surface, COLORS["bridge"], (x, y, CELL, CELL))
        for i in range(1, 4):
            yy = y + i * CELL / 4
            pygame.draw.line(surface, (0, 0, 0, 60), (x, yy), (x + CELL, yy), 1)
    elif terrain == ARTILLERY:
        pygame.draw.circle(surface, COLORS["artillery_body"],
                            (int(x + CELL * 0.5), int(y + CELL * 0.6)), int(CELL * 0.24))
        pygame.draw.line(surface, COLORS["artillery_barrel"],
                          (x + CELL * 0.5, y + CELL * 0.55), (x + CELL * 0.9, y + CELL * 0.2), 4)
    elif terrain == CAMP:
        pygame.draw.rect(surface, COLORS["camp_tent_dark"], (x + 2, y + CELL * 0.55, CELL - 4, CELL * 0.4))
        pygame.draw.polygon(surface, COLORS["camp_tent"], [
            (x + 2, y + CELL * 0.58), (x + CELL / 2, y + CELL * 0.1), (x + CELL - 2, y + CELL * 0.58)
        ])
    elif terrain == RUIN:
        pygame.draw.rect(surface, COLORS["ruin_wall"], (x + 3, y + CELL * 0.3, CELL - 6, CELL * 0.65))
        pygame.draw.line(surface, COLORS["ruin_crack"],
                          (x + CELL * 0.3, y + CELL * 0.3), (x + CELL * 0.45, y + CELL * 0.6), 1)
        pygame.draw.line(surface, COLORS["ruin_crack"],
                          (x + CELL * 0.45, y + CELL * 0.6), (x + CELL * 0.35, y + CELL * 0.95), 1)
    # DIRT: dasar sudah cukup, tidak perlu digambar ulang.


def draw_tank(surface, pos, direction):
    cx, cy = pos[1] * CELL + CELL / 2, pos[0] * CELL + CELL / 2
    theta = math.atan2(direction[0], direction[1])

    pygame.draw.polygon(surface, COLORS["tank"], rotated_rect_points(cx, cy, CELL * 0.64, CELL * 0.48, theta))
    pygame.draw.polygon(surface, COLORS["tank_dark"], rotated_rect_points(cx, cy, CELL * 0.68, CELL * 0.10, theta))
    # trek roda kedua (offset ke arah tegak lurus) digambar lewat translasi kecil
    ox, oy = rotate_point(0, CELL * 0.25, theta)
    pygame.draw.polygon(surface, COLORS["tank_dark"],
                         rotated_rect_points(cx + ox, cy + oy, CELL * 0.68, CELL * 0.10, theta))
    ox2, oy2 = rotate_point(0, -CELL * 0.25, theta)
    pygame.draw.polygon(surface, COLORS["tank_dark"],
                         rotated_rect_points(cx + ox2, cy + oy2, CELL * 0.68, CELL * 0.10, theta))

    pygame.draw.circle(surface, COLORS["tank_dark"], (int(cx), int(cy)), int(CELL * 0.18))
    bx, by = rotate_point(CELL * 0.36, 0, theta)
    pygame.draw.line(surface, COLORS["tank_dark"], (cx, cy), (cx + bx, cy + by), 4)


def draw_drone(surface, pos):
    cx, cy = pos[1] * CELL + CELL / 2, pos[0] * CELL + CELL / 2
    r = CELL * 0.32
    pygame.draw.circle(surface, COLORS["drone"], (int(cx), int(cy)), int(CELL * 0.14))
    for ox, oy in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
        ax, ay = cx + ox * r, cy + oy * r
        pygame.draw.line(surface, COLORS["drone_dark"], (cx, cy), (ax, ay), 2)
        pygame.draw.circle(surface, COLORS["drone_dark"], (int(ax), int(ay)), int(CELL * 0.09))


def draw_vision_ring(surface, pos, vision_radius):
    """Lingkaran transparan menandai jangkauan pandang drone saat ini."""
    cx, cy = pos[1] * CELL + CELL / 2, pos[0] * CELL + CELL / 2
    radius_px = int(vision_radius * CELL)
    ring = pygame.Surface((radius_px * 2 + 4, radius_px * 2 + 4), pygame.SRCALPHA)
    pygame.draw.circle(ring, (*COLORS["vision_ring"], 90), (radius_px + 2, radius_px + 2), radius_px, width=2)
    surface.blit(ring, (cx - radius_px - 2, cy - radius_px - 2))


def draw_overlays(surface, drone):
    """Menggambar jejak pola pencarian (warna tosca redup) dan jalur aktif
    yang sedang diikuti drone saat ini (biru terang saat menyisir, kuning
    krem saat mengejar) — inilah bagian yang membedakan warna pencarian
    preset dari warna jalur kejar sesuai permintaan modifikasi."""
    trail_layer = pygame.Surface((COLS * CELL, ROWS * CELL), pygame.SRCALPHA)
    for (r, c) in drone.visited_cells:
        pygame.draw.rect(trail_layer, (*COLORS["search_trail"], 55),
                          (c * CELL + 6, r * CELL + 6, CELL - 12, CELL - 12))
    surface.blit(trail_layer, (0, 0))

    if drone.current_path:
        active_color = COLORS["chase_path"] if drone.state == Drone.CHASING else COLORS["search_active"]
        path_layer = pygame.Surface((COLS * CELL, ROWS * CELL), pygame.SRCALPHA)
        for step in drone.current_path:
            pygame.draw.rect(path_layer, (*active_color, 170),
                              (step[1] * CELL + 6, step[0] * CELL + 6, CELL - 12, CELL - 12))
        surface.blit(path_layer, (0, 0))


# =========================================================================
# BAGIAN 7 — PANEL INFORMASI (HUD) MEMAKAI FONT PYGAME
# =========================================================================

def draw_text_lines(surface, lines, x, y, font, color, line_height=18):
    for i, line in enumerate(lines):
        surface.blit(font.render(line, True, color), (x, y + i * line_height))
    return y + len(lines) * line_height


def draw_swatch_line(surface, x, y, font, color, label, size=12):
    pygame.draw.rect(surface, color, (x, y + 2, size, size), border_radius=3)
    surface.blit(font.render(label, True, COLORS["text_dim"]), (x + size + 8, y))


# =========================================================================
# BAGIAN 8 — PROGRAM UTAMA (GAME LOOP REAL-TIME)
# =========================================================================
# Berbeda dari versi JS (giliran: tank bergerak -> drone bereaksi sekali),
# versi Python ini berjalan REAL-TIME: tank bisa bergerak terus selama
# tombol ditahan, dan drone melakukan tick pencarian/pengejaran sendiri
# pada interval tetap TANPA menunggu tank bergerak -- sesuai permintaan
# "drone terus melakukan pencarian" karena tanknya juga bergerak terus.

TANK_MOVE_INTERVAL_MS = 140   # jeda antar langkah tank saat tombol ditahan
DRONE_TICK_INTERVAL_MS = 380  # drone mengambil satu langkah tiap interval ini

CONTROLS_TEXT = [
    "Kontrol:",
    " Panah / WASD  : gerak tank",
    " 1 / 2         : algoritma UCS / A*",
    " Z / X / C     : heuristik Manhattan/Euclidean/Chebyshev",
    " [ / ]         : kurangi/tambah radius pandang drone",
    " M             : acak medan baru",
    " R             : reset posisi",
    " E             : eksperimen bandingkan algoritma (lihat konsol)",
    " ESC           : keluar",
]


def run_experiment(grid, drone_pos, tank_pos):
    combos = [
        ("UCS", "ucs", "zero"),
        ("A* + Manhattan", "astar", "manhattan"),
        ("A* + Euclidean", "astar", "euclidean"),
        ("A* + Chebyshev", "astar", "chebyshev"),
    ]
    print("\n=== Eksperimen: drone -> posisi tank saat ini ===")
    print(f"{'Kombinasi':<18}{'Node':>8}{'Biaya':>8}{'ms':>10}")
    for label, algo, heur in combos:
        r = search(grid, drone_pos, tank_pos, algo, heur)
        cost = r["cost"] if r["found"] else "-"
        print(f"{label:<18}{r['expanded']:>8}{str(cost):>8}{r['time_ms']:>10.3f}")
    print("(Manhattan biasanya menang di gerak 4-arah karena paling dekat")
    print(" dengan biaya asli tanpa pernah melebih-lebihkannya / admissible)\n")


def main():
    pygame.init()
    pygame.display.set_caption("Tank vs Drone — UCS/A* (Python)")

    panel_w = 300
    screen = pygame.display.set_mode((COLS * CELL + panel_w, max(ROWS * CELL, 620)))
    clock = pygame.time.Clock()

    font_title = pygame.font.SysFont("segoeui", 18, bold=True)
    font_sub = pygame.font.SysFont("segoeui", 12)
    font = pygame.font.SysFont("consolas", 13)

    def spawn_positions():
        return (ROWS - 2, COLS - 2), (1, 1)

    tank_pos, drone_start = spawn_positions()
    tank_dir = (0, -1)
    grid = generate_map(drone_start, tank_pos)
    drone = Drone(drone_start, grid, vision_radius=4)

    algorithm = "astar"
    heuristic = "manhattan"
    game_over = False
    status_msg = "Drone belum mengetahui posisi tank. Ia mulai menyisir area..."

    last_tank_move = 0
    last_drone_tick = 0

    running = True
    while running:
        now = pygame.time.get_ticks()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_1:
                    algorithm = "ucs"
                elif event.key == pygame.K_2:
                    algorithm = "astar"
                elif event.key == pygame.K_z:
                    heuristic = "manhattan"
                elif event.key == pygame.K_x:
                    heuristic = "euclidean"
                elif event.key == pygame.K_c:
                    heuristic = "chebyshev"
                elif event.key == pygame.K_LEFTBRACKET:
                    drone.vision_radius = max(1, drone.vision_radius - 1)
                elif event.key == pygame.K_RIGHTBRACKET:
                    drone.vision_radius = min(12, drone.vision_radius + 1)
                elif event.key == pygame.K_m:
                    tank_pos, drone_start = spawn_positions()
                    tank_dir = (0, -1)
                    grid = generate_map(drone_start, tank_pos)
                    drone.rebuild_pattern(grid)
                    drone.pos = drone_start
                    game_over = False
                    status_msg = "Medan baru dibuat. Drone mulai menyisir dari nol."
                elif event.key == pygame.K_r:
                    tank_pos, drone_start = spawn_positions()
                    tank_dir = (0, -1)
                    drone.rebuild_pattern(grid)
                    drone.pos = drone_start
                    game_over = False
                    status_msg = "Posisi direset. Drone mulai menyisir dari nol."
                elif event.key == pygame.K_e:
                    run_experiment(grid, drone.pos, tank_pos)

        # --- gerak tank real-time (bisa ditahan tombolnya) ---
        if not game_over and now - last_tank_move >= TANK_MOVE_INTERVAL_MS:
            keys = pygame.key.get_pressed()
            move = None
            if keys[pygame.K_UP] or keys[pygame.K_w]:
                move = (-1, 0)
            elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
                move = (1, 0)
            elif keys[pygame.K_LEFT] or keys[pygame.K_a]:
                move = (0, -1)
            elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                move = (0, 1)

            if move:
                nr, nc = tank_pos[0] + move[0], tank_pos[1] + move[1]
                if 0 <= nr < ROWS and 0 <= nc < COLS and not is_blocked(grid[nr][nc]):
                    tank_pos = (nr, nc)
                    tank_dir = move
                    last_tank_move = now

        # --- tick drone: berjalan sendiri, independen dari gerak tank ---
        if not game_over and now - last_drone_tick >= DRONE_TICK_INTERVAL_MS:
            last_drone_tick = now
            drone.take_turn(grid, tank_pos, algorithm, heuristic)

            if drone.pos == tank_pos:
                game_over = True
                status_msg = "Tank tertangkap! Tekan R untuk reset posisi."
            elif drone.state == Drone.CHASING:
                status_msg = "Tank terdeteksi! Drone mengejar dan terus menghitung ulang jalurnya..."
            else:
                status_msg = "Tank di luar jangkauan pandang. Drone melanjutkan pola pencarian..."

        # ---------------- RENDER ----------------
        screen.fill(COLORS["bg_deep"])

        board = screen.subsurface((0, 0, COLS * CELL, ROWS * CELL))
        for r in range(ROWS):
            for c in range(COLS):
                draw_cell(board, grid[r][c], r, c)

        draw_overlays(board, drone)
        if drone.state == Drone.SEARCHING:
            draw_vision_ring(board, drone.pos, drone.vision_radius)
        draw_tank(board, tank_pos, tank_dir)
        draw_drone(board, drone.pos)

        # ---- panel kanan ----
        px = COLS * CELL + 16
        y = 14
        screen.blit(font_title.render("Tank vs Drone — Python", True, COLORS["text_cream"]), (px, y))
        y += 30
        y = draw_text_lines(screen, [
            "Drone tidak tahu posisi tank di",
            "awal. Ia menyisir medan sampai",
            "tank masuk radius pandangnya.",
        ], px, y, font_sub, COLORS["text_dim"], 15) + 8

        y = draw_text_lines(screen, ["Legenda:"], px, y, font, COLORS["text_cream"], 16)
        legend = [
            (COLORS["tank"], "Tank (pemain)"),
            (COLORS["drone"], "Drone (pengejar)"),
            (COLORS["chase_path"], "Jalur kejar (tahu posisi tank)"),
            (COLORS["search_active"], "Jalur pencarian (belum tahu)"),
            (COLORS["search_trail"], "Jejak area yang sudah disisir"),
            (COLORS["tree_canopy"], "Pohon"),
            (COLORS["rock"], "Bebatuan"),
            (COLORS["river"], "Sungai"),
            (COLORS["bridge"], "Jembatan (biaya 3)"),
            (COLORS["artillery_body"], "Artileri musuh"),
            (COLORS["camp_tent"], "Kamp tentara"),
            (COLORS["ruin_wall"], "Bangunan runtuh"),
        ]
        for color, label in legend:
            draw_swatch_line(screen, px, y, font, color, label)
            y += 18
        y += 8

        y = draw_text_lines(screen, CONTROLS_TEXT, px, y, font, COLORS["text_cream"], 16) + 8

        algo_label = "UCS" if algorithm == "ucs" else f"A* ({heuristic})"
        stats = drone.last_stats
        info_lines = [
            f"Algoritma   : {algo_label}",
            f"Radius liat : {drone.vision_radius} sel",
            f"Status drone: {drone.state}",
        ]
        if stats:
            cost_txt = stats["cost"] if stats["found"] else "-"
            info_lines += [
                f"Node dieksp.: {stats['expanded']}",
                f"Biaya jalur : {cost_txt}",
                f"Waktu       : {stats['time_ms']:.3f} ms",
            ]
        y = draw_text_lines(screen, info_lines, px, y, font, COLORS["text_cream"], 16) + 8

        status_color = COLORS["danger"] if game_over else COLORS["text_dim"]
        wrapped = [status_msg[i:i + 34] for i in range(0, len(status_msg), 34)]
        draw_text_lines(screen, wrapped, px, y, font_sub, status_color, 15)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
