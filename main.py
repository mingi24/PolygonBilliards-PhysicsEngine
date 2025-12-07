import pygame
import numpy as np
import math
import random

# --- 설정 및 상수 ---
SCREEN_WIDTH, SCREEN_HEIGHT = 1000, 600
FPS = 60

# 당구대 설정
TABLE_WIDTH, TABLE_HEIGHT = 800, 400
TABLE_X = (SCREEN_WIDTH - TABLE_WIDTH) / 2
TABLE_Y = (SCREEN_HEIGHT - TABLE_HEIGHT) / 2
BORDER_SIZE = 30
HOLE_RADIUS = 25

GREEN_TABLE = (34, 139, 34)
BROWN_BORDER = (139, 69, 19)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (200, 200, 200)
RED = (255, 50, 50)
YELLOW = (255, 215, 0)
BUTTON_COLOR = (70, 130, 180)
HOVER_COLOR = (100, 149, 237)

class RigidBody:
    def __init__(self, x, y, mass, color, shape_type="poly", size=30):
        self.pos = np.array([float(x), float(y)])
        self.vel = np.array([0.0, 0.0])
        self.angle = 0.0
        self.ang_vel = 0.0 
        
        self.mass = mass
        self.inv_mass = 1.0 / mass if mass > 0 else 0.0
        self.color = color
        self.size = size
        self.is_cue = (shape_type == "cue")

        radius = size / 2.0
        self.inertia = 0.5 * mass * (radius ** 2)
        self.inv_inertia = 1.0 / self.inertia if self.inertia > 0 else 0.0

        self.local_vertices = []
        if self.is_cue: 
            sides = 16 
        else:
            sides = random.randint(3, 6)
            
        for i in range(sides):
            theta = 2 * math.pi * i / sides
            r = size / 2.0
            self.local_vertices.append(np.array([r * math.cos(theta), r * math.sin(theta)]))

    def update(self, dt):
        self.pos += self.vel * dt
        self.angle += self.ang_vel * dt

        self.vel *= 0.985
        self.ang_vel *= 0.98

        if np.linalg.norm(self.vel) < 2.0: self.vel[:] = 0
        if abs(self.ang_vel) < 0.1: self.ang_vel = 0

        e_wall = 0.8
        if self.pos[0] < TABLE_X + self.size/2:
            self.pos[0] = TABLE_X + self.size/2
            self.vel[0] *= -e_wall
        elif self.pos[0] > TABLE_X + TABLE_WIDTH - self.size/2:
            self.pos[0] = TABLE_X + TABLE_WIDTH - self.size/2
            self.vel[0] *= -e_wall
            
        if self.pos[1] < TABLE_Y + self.size/2:
            self.pos[1] = TABLE_Y + self.size/2
            self.vel[1] *= -e_wall
        elif self.pos[1] > TABLE_Y + TABLE_HEIGHT - self.size/2:
            self.pos[1] = TABLE_Y + TABLE_HEIGHT - self.size/2
            self.vel[1] *= -e_wall

    def get_world_vertices(self):
        c, s = math.cos(self.angle), math.sin(self.angle)
        rot_matrix = np.array([[c, -s], [s, c]])
        return [self.pos + rot_matrix @ v for v in self.local_vertices]

# --- 충돌 감지 (SAT) ---

def get_axes(vertices):
    axes = []
    for i in range(len(vertices)):
        p1, p2 = vertices[i], vertices[(i + 1) % len(vertices)]
        edge = p1 - p2
        normal = np.array([-edge[1], edge[0]])
        length = np.linalg.norm(normal)
        if length > 0: axes.append(normal / length)
    return axes

def project(vertices, axis):
    dots = [np.dot(v, axis) for v in vertices]
    return min(dots), max(dots)

def check_collision_sat(body_a, body_b):
    verts_a = body_a.get_world_vertices()
    verts_b = body_b.get_world_vertices()
    
    axes = get_axes(verts_a) + get_axes(verts_b)
    
    min_overlap = float('inf')
    collision_normal = None
    
    for axis in axes:
        min_a, max_a = project(verts_a, axis)
        min_b, max_b = project(verts_b, axis)
        
        if max_a < min_b or max_b < min_a:
            return False, None, 0
        
        overlap = min(max_a, max_b) - max(min_a, min_b)
        if overlap < min_overlap:
            min_overlap = overlap
            collision_normal = axis

    if np.dot(body_b.pos - body_a.pos, collision_normal) < 0:
        collision_normal = -collision_normal
        
    return True, collision_normal, min_overlap

# --- 충돌 반응 (Impulse) ---

def get_support(vertices, direction):
    best_projection = -float('inf')
    best_point = None
    for v in vertices:
        projection = np.dot(v, direction)
        if projection > best_projection:
            best_projection = projection
            best_point = v
    return best_point

def resolve_collision(body_a, body_b, normal, depth):
    total_inv = body_a.inv_mass + body_b.inv_mass
    if total_inv == 0: return
    
    move = normal * (depth * 0.5)
    body_a.pos -= move
    body_b.pos += move

    contact_a = get_support(body_a.get_world_vertices(), normal)
    contact_b = get_support(body_b.get_world_vertices(), -normal)

    vals_a = np.dot(contact_a - body_b.pos, -normal)
    vals_b = np.dot(contact_b - body_a.pos, normal)
    if vals_a > vals_b:
        contact_point = contact_a
    else:
        contact_point = contact_b

    r_a = contact_point - body_a.pos
    r_b = contact_point - body_b.pos

    def cross_2d(v1, v2): return v1[0]*v2[1] - v1[1]*v2[0]

    vel_a = body_a.vel + np.array([-body_a.ang_vel * r_a[1], body_a.ang_vel * r_a[0]])
    vel_b = body_b.vel + np.array([-body_b.ang_vel * r_b[1], body_b.ang_vel * r_b[0]])
    rel_vel = vel_b - vel_a
    vel_normal = np.dot(rel_vel, normal)

    if vel_normal > 0: return

    e = 0.9 
    rn_a = cross_2d(r_a, normal)
    rn_b = cross_2d(r_b, normal)
    
    denom = (total_inv + rn_a**2 * body_a.inv_inertia + rn_b**2 * body_b.inv_inertia)
    j = -(1 + e) * vel_normal / denom
    
    impulse = j * normal
    
    body_a.vel -= impulse * body_a.inv_mass
    body_a.ang_vel -= cross_2d(r_a, impulse) * body_a.inv_inertia
    
    body_b.vel += impulse * body_b.inv_mass
    body_b.ang_vel += cross_2d(r_b, impulse) * body_b.inv_inertia

# --- 게임 로직 및 UI ---

def get_pockets():
    xs = [TABLE_X + BORDER_SIZE, TABLE_X + TABLE_WIDTH/2, TABLE_X + TABLE_WIDTH - BORDER_SIZE]
    ys = [TABLE_Y + BORDER_SIZE, TABLE_Y + TABLE_HEIGHT - BORDER_SIZE]
    return [np.array([x, y]) for y in ys for x in xs]

def check_pocket_fall(ball, pockets):
    for pocket in pockets:
        if np.linalg.norm(ball.pos - pocket) < HOLE_RADIUS:
            return True
    return False

def init_game():
    balls = []
    cue_ball = RigidBody(TABLE_X + 200, TABLE_Y + TABLE_HEIGHT/2, 20, WHITE, "cue", 35)
    balls.append(cue_ball)
    
    start_x = TABLE_X + 600
    start_y = TABLE_Y + TABLE_HEIGHT/2
    for i in range(5):
        for j in range(i + 1):
            x = start_x + i * 38
            y = start_y + (j * 38) - (i * 19)
            color = (random.randint(50, 200), random.randint(50, 200), random.randint(50, 255))
            balls.append(RigidBody(x, y, 20, color, "poly", 35))
    return balls, cue_ball

def draw_button(screen, rect, text, font, is_hovered):
    color = HOVER_COLOR if is_hovered else BUTTON_COLOR
    pygame.draw.rect(screen, color, rect, border_radius=5)
    pygame.draw.rect(screen, WHITE, rect, 2, border_radius=5)
    text_surf = font.render(text, True, WHITE)
    text_rect = text_surf.get_rect(center=rect.center)
    screen.blit(text_surf, text_rect)

# --- 메인 실행 ---

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Project #3: Polygon Billiards Final")
    clock = pygame.time.Clock()
    
    font_ui = pygame.font.SysFont("Arial", 20)
    font_big = pygame.font.SysFont("Arial", 60, bold=True)

    balls, cue_ball = init_game()
    pockets = get_pockets()
    
    is_dragging = False
    start_drag = None
    show_menu = False
    
    btn_menu_rect = pygame.Rect(SCREEN_WIDTH - 110, 10, 100, 40)
    btn_reset_rect = pygame.Rect(SCREEN_WIDTH - 110, 60, 100, 40)
    btn_quit_rect = pygame.Rect(SCREEN_WIDTH - 110, 110, 100, 40)

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0
        mouse_pos = pygame.mouse.get_pos()

        # 이벤트 처리
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    if btn_menu_rect.collidepoint(mouse_pos):
                        show_menu = not show_menu
                    elif show_menu:
                        if btn_reset_rect.collidepoint(mouse_pos):
                            balls, cue_ball = init_game()
                            show_menu = False
                        elif btn_quit_rect.collidepoint(mouse_pos):
                            running = False
                        else: show_menu = False
                    elif not show_menu and len(balls) > 1:
                        if np.linalg.norm(cue_ball.vel) < 5:
                            is_dragging = True
                            start_drag = mouse_pos

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1 and is_dragging:
                    end_drag = mouse_pos
                    force = np.array(start_drag) - np.array(end_drag)
                    if np.linalg.norm(force) > 300:
                        force = force / np.linalg.norm(force) * 300
                    cue_ball.vel = force * 5.0
                    is_dragging = False

        # --- 물리 ---
        balls_to_remove = []
        for ball in balls:
            ball.update(dt)

            if check_pocket_fall(ball, pockets):
                if ball.is_cue:
                    ball.pos = np.array([TABLE_X + 200, TABLE_Y + TABLE_HEIGHT/2])
                    ball.vel[:] = 0
                    ball.ang_vel = 0
                else:
                    balls_to_remove.append(ball)
        
        for b in balls_to_remove:
            if b in balls: balls.remove(b)

        # 충돌 처리
        for i in range(len(balls)):
            for j in range(i + 1, len(balls)):
                is_col, norm, depth = check_collision_sat(balls[i], balls[j])
                if is_col: resolve_collision(balls[i], balls[j], norm, depth)

        # --- 렌더링 ---
        screen.fill((50, 30, 10))
        
        pygame.draw.rect(screen, BROWN_BORDER, (TABLE_X-BORDER_SIZE, TABLE_Y-BORDER_SIZE, TABLE_WIDTH+BORDER_SIZE*2, TABLE_HEIGHT+BORDER_SIZE*2), border_radius=10)
        pygame.draw.rect(screen, GREEN_TABLE, (TABLE_X, TABLE_Y, TABLE_WIDTH, TABLE_HEIGHT))

        for p in pockets:
            pygame.draw.circle(screen, BLACK, (int(p[0]), int(p[1])), HOLE_RADIUS)

        if is_dragging and len(balls) > 1:
            pygame.draw.line(screen, WHITE, cue_ball.pos, mouse_pos, 2)

        for ball in balls:
            verts = ball.get_world_vertices()
            pygame.draw.polygon(screen, ball.color, verts)
            pygame.draw.polygon(screen, BLACK, verts, 2)

        # --- UI ---
        remaining = len(balls) - 1
        
        if remaining == 0:
            overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            screen.blit(overlay, (0,0))
            win_txt = font_big.render("CONGRATULATIONS!", True, YELLOW)
            screen.blit(win_txt, win_txt.get_rect(center=(SCREEN_WIDTH/2, SCREEN_HEIGHT/2 - 20)))
            sub_txt = font_ui.render("Press 'MENU' -> 'RESET' to play again", True, WHITE)
            screen.blit(sub_txt, sub_txt.get_rect(center=(SCREEN_WIDTH/2, SCREEN_HEIGHT/2 + 40)))
        else:
            screen.blit(font_ui.render(f"Remaining Balls: {remaining}", True, WHITE), (20, 20))

        draw_button(screen, btn_menu_rect, "MENU", font_ui, btn_menu_rect.collidepoint(mouse_pos))
        if show_menu:
            draw_button(screen, btn_reset_rect, "RESET", font_ui, btn_reset_rect.collidepoint(mouse_pos))
            draw_button(screen, btn_quit_rect, "QUIT", font_ui, btn_quit_rect.collidepoint(mouse_pos))

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()