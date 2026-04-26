import os
import csv
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import pygame
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# Opcional: para graficar los datos en 2D y 3D
import matplotlib
# Configuramos backend para ventanas interactivas (TkAgg funciona en la mayoría de sistemas)
try:
    matplotlib.use("TkAgg")
except Exception:
    try:
        matplotlib.use("Qt5Agg")
    except Exception:
        pass  # Usa el backend por defecto
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401, necesario para activar 3D en matplotlib

# Activamos modo interactivo para que las ventanas no bloqueen el juego
plt.ion()


# Ventana base y factor de escala
BASE_W, BASE_H = 1080, 720
WINDOW_FRACTION = 0.97
EXTRA_SCALE = 1.1

# Acciones del modelo (multiclase)
ACCION_NADA    = 0
ACCION_SALTO   = 1
ACCION_AGACHAR = 2

# Tipo de bala
BALA_BAJA = 0   # al nivel del suelo  → esquivar saltando
BALA_ALTA = 1   # a media altura      → esquivar agachándose


@dataclass
class Sample:
    velocidad_bala: float
    distancia: float
    tipo_bala: int    # BALA_BAJA o BALA_ALTA
    accion: int       # ACCION_NADA / ACCION_SALTO / ACCION_AGACHAR


class Juego:
    def __init__(self) -> None:
        pygame.init()

        self._flags = 0
        self._fullscreen = False

        start_w = BASE_W
        start_h = BASE_H
        self.pantalla = pygame.display.set_mode((start_w, start_h), self._flags)
        pygame.display.set_caption("Juego: Bala + salto + agacharse + MLP")

        # Colores
        self.BLANCO  = (255, 255, 255)
        self.NEGRO   = (0,   0,   0)
        self.GRIS    = (200, 200, 200)
        self.AMARILLO = (255, 220, 120)
        self.CYAN    = (100, 220, 255)

        # Estado global
        self.corriendo = True
        self.modo_auto = False

        # Datos / modelo
        self.datos_modelo: List[Sample] = []
        self.modelo: Optional[MLPClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        self.modelo_entrenado = False
        self.clase_unica: Optional[int] = None
        self.ultima_proba: Optional[list] = None   # [p_nada, p_salto, p_agachar]

        # Parámetros de decisión
        self.decision_window = 500
        self.decision_record_every = 3
        self._decision_frame_counter = 0

        # Geometría / física (se rellenan en _apply_resolution)
        self.w, self.h = start_w, start_h
        self.scale = 1.0
        self.margin = 50
        self.ground_y = self.h - 100
        self.player_size       = (32, 48)
        self.player_size_agach = (32, 24)   # hitbox reducida al agacharse
        self.bullet_size = (16, 16)
        self.ship_size   = (64, 64)
        self.fondo_speed = 3

        # Estado de salto
        self.salto = False
        self.en_suelo = True
        self.salto_vel_inicial = 15.0
        self.gravedad  = 1.0
        self.salto_vel = self.salto_vel_inicial

        # Estado de agacharse
        self.agachado = False
        self.agachado_timer = 0
        self.AGACHADO_FRAMES = 25   # cuántos frames dura el agache

        # Animación del personaje
        self.current_frame = 0
        self.frame_speed   = 10
        self.frame_count   = 0

        # Animación del agachado
        self.agach_frame       = 0
        self.agach_frame_count = 0
        self.agach_frame_speed = 4

        # Bala
        self.velocidad_bala = -12
        self.bala_disparada = False
        self.tipo_bala_actual = BALA_BAJA

        self.fondo_x1 = 0
        self.fondo_x2 = start_w

        self._apply_resolution(start_w, start_h, reset_positions=True)
        self._reset_estado_juego()

    # ----------------- resolución / assets -----------------
    def _apply_resolution(self, w: int, h: int, reset_positions: bool) -> None:
        self.w, self.h = int(w), int(h)

        self.scale = min(self.w / BASE_W, self.h / BASE_H) * EXTRA_SCALE
        self.scale = max(1.0, self.scale)

        self.margin       = int(50 * self.scale)
        ground_offset     = int(100 * self.scale)
        self.ground_y     = self.h - ground_offset

        self.player_size       = (int(32 * self.scale), int(48 * self.scale))
        self.player_size_agach = (int(32 * self.scale), int(24 * self.scale))
        self.bullet_size = (int(16 * self.scale), int(16 * self.scale))
        self.ship_size   = (int(64 * self.scale), int(64 * self.scale))
        self.fondo_speed = max(1, int(2 * self.scale))

        self.salto_vel_inicial = 12 * self.scale
        self.gravedad  = 1  * self.scale
        self.salto_vel = self.salto_vel_inicial

        self.AGACHADO_FRAMES = 25

        self.decision_window = int(500 * self.scale)

        self.fuente       = pygame.font.SysFont("Arial", int(24 * self.scale))
        self.fuente_chica = pygame.font.SysFont("Arial", int(18 * self.scale))

        self._cargar_assets()

        if reset_positions or not hasattr(self, "jugador"):
            self.jugador = pygame.Rect(
                self.margin, self.ground_y,
                self.player_size[0], self.player_size[1]
            )
            self.bala = pygame.Rect(
                self.w - self.margin,
                self.ground_y + int(10 * self.scale),
                self.bullet_size[0], self.bullet_size[1],
            )
            self.nave = pygame.Rect(
                self.w - int(100 * self.scale),
                self.ground_y,
                self.ship_size[0], self.ship_size[1],
            )

    def _cargar_assets(self) -> None:
        def safe_load(path: str, size: Tuple[int, int], fallback_color=(200, 200, 200, 255)) -> pygame.Surface:
            try:
                img = pygame.image.load(path).convert_alpha()
                return pygame.transform.smoothscale(img, size)
            except Exception:
                surf = pygame.Surface(size, pygame.SRCALPHA)
                surf.fill(fallback_color)
                return surf

        base = os.path.dirname(__file__)
        self.jugador_frames = [
            safe_load(os.path.join(base, "assets/sprites/sonic1.png"), self.player_size),
            safe_load(os.path.join(base, "assets/sprites/sonic2.png"), self.player_size),
            safe_load(os.path.join(base, "assets/sprites/sonic3.png"), self.player_size),
            safe_load(os.path.join(base, "assets/sprites/sonic4.png"), self.player_size),
        ]
        # Frame agachado: recortamos el sprite normal a la mitad inferior
        self.jugador_frames_agach = [
            safe_load(os.path.join(base, "assets/sprites/spin1.png"), self.player_size_agach),
            safe_load(os.path.join(base, "assets/sprites/spin2.png"), self.player_size_agach),
            safe_load(os.path.join(base, "assets/sprites/spin3.png"), self.player_size_agach),
            safe_load(os.path.join(base, "assets/sprites/spin4.png"), self.player_size_agach),
        ]

        self.bala_img = safe_load(
            os.path.join(base, "assets/sprites/purple_ball.png"),
            self.bullet_size,
            (160, 120, 255, 255),
        )
        # Bala alta: color diferente para distinguirla visualmente
        self.bala_alta_img = safe_load(
            os.path.join(base, "assets/sprites/purple_ball.png"),
            self.bullet_size,
            (255, 120, 120, 255),
        )
        self.fondo_img = safe_load(
            os.path.join(base, "assets/game/fondo2.png"),
            (self.w, self.h),
            (40, 40, 40, 255),
        )
        self.nave_img = safe_load(
            os.path.join(base, "assets/game/ufo.png"),
            self.ship_size,
            (140, 255, 200, 255),
        )

    def _toggle_fullscreen(self) -> None:
        self._fullscreen = not self._fullscreen
        if self._fullscreen:
            info = pygame.display.Info()
            w = info.current_w or self.w
            h = info.current_h or self.h
            self.pantalla = pygame.display.set_mode((w, h), pygame.FULLSCREEN)
            self._apply_resolution(w, h, reset_positions=True)
        else:
            self.pantalla = pygame.display.set_mode((BASE_W, BASE_H), self._flags)
            self._apply_resolution(BASE_W, BASE_H, reset_positions=True)
        self._reset_estado_juego()

    # ----------------- estado juego / modelo -----------------
    def _reset_estado_juego(self) -> None:
        self.jugador.x, self.jugador.y = self.margin, self.ground_y
        self.jugador.width, self.jugador.height = self.player_size
        self.nave.x, self.nave.y = self.w - int(100 * self.scale), self.ground_y
        self.bala.x = self.w - self.margin
        self.bala.y = self.ground_y + int(10 * self.scale)
        self.bala_disparada = False
        self.velocidad_bala = int(-10 * self.scale)
        self.tipo_bala_actual = BALA_BAJA
        self.salto    = False
        self.en_suelo = True
        self.salto_vel = self.salto_vel_inicial
        self.agachado = False
        self.agachado_timer = 0
        self._decision_frame_counter = 0
        self.fondo_x1 = 0
        self.fondo_x2 = self.w

    def _reset_modelo(self) -> None:
        self.modelo = None
        self.scaler = None
        self.modelo_entrenado = False
        self.clase_unica = None

    # ----------------- export / gráficas -----------------
    def exportar_datos_csv(self) -> str:
        if not self.datos_modelo:
            return "No hay datos para exportar."

        base = os.path.dirname(__file__)
        ruta = os.path.join(base, "datos_mlp1.csv")

        try:
            with open(ruta, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["velocidad_bala", "distancia", "tipo_bala", "accion"])
                for s in self.datos_modelo:
                    writer.writerow([s.velocidad_bala, s.distancia, s.tipo_bala, s.accion])
        except Exception as e:
            return f"Error al guardar CSV: {e}"

        return f"CSV guardado en datos_mlp1.csv ({len(self.datos_modelo)} filas)."

    def graficar_datos_2d(self) -> str:
        if not self.datos_modelo:
            return "No hay datos para graficar."

        xs = [s.distancia      for s in self.datos_modelo]
        ys = [s.velocidad_bala for s in self.datos_modelo]
        colores = {
            ACCION_NADA:    "blue",
            ACCION_SALTO:   "red",
            ACCION_AGACHAR: "green",
        }
        cs = [colores.get(s.accion, "gray") for s in self.datos_modelo]

        fig_num = plt.figure("Datos MLP - 2D", figsize=(8, 6)).number
        plt.figure(fig_num)
        plt.clf()
        ax = plt.gca()
        ax.scatter(xs, ys, c=cs, alpha=0.6, edgecolors="k", s=30)
        ax.set_xlabel("Distancia jugador-bala")
        ax.set_ylabel("Velocidad bala")
        ax.set_title("Datos MLP (azul=nada, rojo=salto, verde=agachar)")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show(block=False)
        plt.draw()
        return "Mostrando gráfica 2D interactiva."

    def graficar_datos_3d(self) -> str:
        if not self.datos_modelo:
            return "No hay datos para graficar."

        xs = [s.distancia      for s in self.datos_modelo]
        ys = [s.velocidad_bala for s in self.datos_modelo]
        zs = list(range(len(self.datos_modelo)))
        colores = {
            ACCION_NADA:    "blue",
            ACCION_SALTO:   "red",
            ACCION_AGACHAR: "green",
        }
        cs = [colores.get(s.accion, "gray") for s in self.datos_modelo]

        fig = plt.figure("Datos MLP - 3D", figsize=(8, 6))
        plt.clf()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(xs, ys, zs, c=cs, alpha=0.6, edgecolors="k", s=30)
        ax.set_xlabel("Distancia")
        ax.set_ylabel("Velocidad bala")
        ax.set_zlabel("Índice (tiempo aproximado)")
        ax.set_title("Datos MLP 3D (azul=nada, rojo=salto, verde=agachar)")
        plt.tight_layout()
        plt.show(block=False)
        plt.draw()
        return "Mostrando gráfica 3D interactiva."

    # ----------------- bala / salto / agacharse -----------------
    def disparar_bala(self) -> None:
        if not self.bala_disparada:
            self.velocidad_bala = int(random.randint(-12, -6) * self.scale)
            self.bala_disparada = True

            # Decidir tipo de bala aleatoriamente
            self.tipo_bala_actual = random.choice([BALA_BAJA, BALA_ALTA])

            if self.tipo_bala_actual == BALA_BAJA:
                # Bala a nivel del suelo (igual que antes)
                self.bala.y = self.ground_y + int(18 * self.scale)
            else:
                # Bala alta: a media altura del personaje (hay que agacharse)
                self.bala.y = self.ground_y - int(self.player_size[1] * 0.22)

            self.bala.x = self.w - self.margin

    def reset_bala(self) -> None:
        self.bala.x = self.w - self.margin
        self.bala_disparada = False
        self.tipo_bala_actual = BALA_BAJA

    def iniciar_salto(self) -> None:
        if self.en_suelo and not self.agachado:
            self.salto    = True
            self.en_suelo = False

    def manejar_salto(self) -> None:
        if self.salto:
            self.jugador.y -= int(self.salto_vel)
            self.salto_vel -= self.gravedad
            if self.jugador.y >= self.ground_y:
                self.jugador.y = self.ground_y
                self.salto     = False
                self.salto_vel = self.salto_vel_inicial
                self.en_suelo  = True

    def iniciar_agache(self) -> None:
        """Activa el agache si el jugador está en el suelo y no está saltando."""
        if self.en_suelo and not self.salto:
            if not self.agachado:
                self.agachado       = True
                self.agachado_timer = self.AGACHADO_FRAMES
                # Reducir hitbox: el jugador "se encoge" hacia abajo
                self.jugador.height = self.player_size_agach[1]
                self.jugador.y      = self.ground_y + (self.player_size[1] - self.player_size_agach[1])

    def manejar_agache(self) -> None:
        """Decrementa el timer del agache y restaura el tamaño cuando termina."""
        if self.agachado:
            self.agachado_timer -= 1
            if self.agachado_timer <= 0:
                self.agachado       = False
                self.jugador.height = self.player_size[1]
                self.jugador.y      = self.ground_y

    # ----------------- datos / ML -----------------
    def registrar_decision_manual(self) -> None:
        if not self.bala_disparada:
            return
        distancia = abs(self.jugador.x - self.bala.x)

        if self.agachado:
            accion = ACCION_AGACHAR
        elif not self.en_suelo:
            accion = ACCION_SALTO
        else:
            accion = ACCION_NADA

        self.datos_modelo.append(
            Sample(
                velocidad_bala=float(self.velocidad_bala),
                distancia=float(distancia),
                tipo_bala=int(self.tipo_bala_actual),
                accion=accion,
            )
        )

    def entrenar_modelo(self) -> Tuple[bool, str]:
        samples = list(self.datos_modelo)
        if len(samples) < 80:
            return False, "Necesitas más datos (>= 80). Juega en MANUAL."

        # Features: velocidad_bala, distancia, tipo_bala
        X = [[s.velocidad_bala, s.distancia, float(s.tipo_bala)] for s in samples]
        y = [s.accion for s in samples]

        clases = sorted(set(y))
        if len(clases) < 2:
            self._reset_modelo()
            self.clase_unica = int(clases[0])
            self.modelo_entrenado = True
            nombres = {0: "NADA", 1: "SALTO", 2: "AGACHAR"}
            return True, f"Modelo trivial: siempre {nombres.get(self.clase_unica,'?')}. Juega más para obtener variedad."

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        clf = MLPClassifier(
            hidden_layer_sizes=(16, 8),   # un poco más de capacidad para 3 clases
            activation="relu",
            solver="adam",
            max_iter=300000,
            random_state=42,
        )
        clf.fit(X_train, y_train)
        acc = clf.score(X_test, y_test)

        self._reset_modelo()
        self.scaler = scaler
        self.modelo = clf
        self.modelo_entrenado = True
        return True, f"MLP entrenado (3 clases). Accuracy test ≈ {acc:.3f}"

    def decision_auto(self) -> int:
        """
        Devuelve la acción que el modelo decide para este frame.
        ACCION_NADA / ACCION_SALTO / ACCION_AGACHAR
        """
        if not self.modelo_entrenado:
            return ACCION_NADA
        if not self.bala_disparada:
            return ACCION_NADA

        distancia = abs(self.jugador.x - self.bala.x)

        # Caso trivial (una sola clase observada)
        if self.clase_unica is not None and self.modelo is None:
            return self.clase_unica

        if self.modelo is None or self.scaler is None:
            return ACCION_NADA

        X  = [[float(self.velocidad_bala), float(distancia), float(self.tipo_bala_actual)]]
        Xs = self.scaler.transform(X)

        if hasattr(self.modelo, "predict_proba"):
            probas = self.modelo.predict_proba(Xs)[0]
            # Mapear probas a las clases reales que el modelo conoce
            clases = list(self.modelo.classes_)
            proba_vec = [0.0, 0.0, 0.0]
            for i, c in enumerate(clases):
                if c < 3:
                    proba_vec[c] = float(probas[i])
            self.ultima_proba = proba_vec
            accion = int(clases[probas.argmax()])
        else:
            accion = int(self.modelo.predict(Xs)[0])
            self.ultima_proba = None

        return accion

    # ----------------- menú -----------------
    def _dibujar_menu(self, msg: str = "") -> None:
        self.pantalla.fill(self.NEGRO)
        titulo = self.fuente.render("MENÚ", True, self.BLANCO)
        self.pantalla.blit(titulo, (self.w // 2 - titulo.get_width() // 2, int(60 * self.scale)))

        opciones = [
            "M - Manual (reinicia dataset y borra modelo)",
            "A - Auto (usa MLP; sin modelo NO actúa)",
            "T - Entrenar MLP",
            "C - Exportar datos a CSV",
            "F - Fullscreen (toggle)",
            "Q - Salir",
            "",
            "Controles en juego:",
            "  ESPACIO      → saltar   (esquiva bala BAJA)",
            "  ABAJO / S    → agacharse (esquiva bala ALTA)",
        ]
        x0     = int(80 * self.scale)
        y      = int(140 * self.scale)
        line_h = self.fuente.get_linesize()
        pad    = max(6, int(6 * self.scale))
        for op in opciones:
            fuente = self.fuente_chica if op.startswith("  ") else self.fuente
            color  = self.CYAN if op.startswith("  ") else self.BLANCO
            t = fuente.render(op, True, color)
            self.pantalla.blit(t, (x0, y))
            y += (self.fuente_chica.get_linesize() if op.startswith("  ") else line_h) + pad

        y += int(8 * self.scale)
        estado = [
            f"Memoria: {len(self.datos_modelo)} | Modelo: {'sí' if self.modelo_entrenado else 'no'}",
            f"Resolución: {self.w}x{self.h} | scale≈{self.scale:.2f}",
        ]
        for line in estado:
            t = self.fuente_chica.render(line, True, self.GRIS)
            self.pantalla.blit(t, (x0, y))
            y += self.fuente_chica.get_linesize()

        if msg:
            mm = self.fuente_chica.render(msg, True, self.AMARILLO)
            self.pantalla.blit(mm, (x0, y + int(12 * self.scale)))

        pygame.display.flip()

    def mostrar_menu(self) -> None:
        msg = ""
        esperando = True
        self._decision_frame_counter = 0
        while esperando and self.corriendo:
            self._dibujar_menu(msg)
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    self.corriendo = False
                    esperando = False
                    break
                if e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_m:
                        self.modo_auto = False
                        self.datos_modelo.clear()
                        self._reset_modelo()
                        self._reset_estado_juego()
                        esperando = False
                        break
                    if e.key == pygame.K_a:
                        if not self.modelo_entrenado:
                            msg = "Primero entrena el MLP (T) en esta sesión."
                        else:
                            self.modo_auto = True
                            self._reset_estado_juego()
                            esperando = False
                            break
                    if e.key == pygame.K_t:
                        ok, info = self.entrenar_modelo()
                        msg = info if ok else f"Error: {info}"
                    if e.key == pygame.K_c:
                        msg = self.exportar_datos_csv()
                    if e.key == pygame.K_f:
                        self._toggle_fullscreen()
                    if e.key == pygame.K_q:
                        self.corriendo = False
                        esperando = False
                        return

    # ----------------- render / loop -----------------
    def _update_frame(self) -> None:
        # Fondo
        self.fondo_x1 -= self.fondo_speed
        self.fondo_x2 -= self.fondo_speed
        if self.fondo_x1 <= -self.w:
            self.fondo_x1 = self.w
        if self.fondo_x2 <= -self.w:
            self.fondo_x2 = self.w
        self.pantalla.blit(self.fondo_img, (self.fondo_x1, 0))
        self.pantalla.blit(self.fondo_img, (self.fondo_x2, 0))

        # Animación del jugador
        self.frame_count += 1
        if self.frame_count >= self.frame_speed:
            self.current_frame = (self.current_frame + 1) % len(self.jugador_frames)
            self.frame_count   = 0

        if self.agachado:
            self.agach_frame_count += 1
            if self.agach_frame_count >= self.agach_frame_speed:
                self.agach_frame = (self.agach_frame + 1) % len(self.jugador_frames_agach)
                self.agach_frame_count = 0
            self.pantalla.blit(self.jugador_frames_agach[self.agach_frame], (self.jugador.x, self.jugador.y))
        else:
            self.pantalla.blit(self.jugador_frames[self.current_frame], (self.jugador.x, self.jugador.y))

        self.pantalla.blit(self.nave_img, (self.nave.x, self.nave.y))

        # Bala
        if self.bala_disparada:
            self.bala.x += self.velocidad_bala
        if self.bala.x < -self.bullet_size[0]:
            self.reset_bala()

        # Dibujar bala con color según tipo
        bala_img = self.bala_alta_img if self.tipo_bala_actual == BALA_ALTA else self.bala_img
        self.pantalla.blit(bala_img, (self.bala.x, self.bala.y))

        # Colisión
        if self.jugador.colliderect(self.bala):
            self._reset_estado_juego()

        # HUD en tiempo real
        self._dibujar_hud()

    def _dibujar_hud(self) -> None:
        """Muestra información del modelo y controles en pantalla."""
        y = 10
        if self.modelo_entrenado and self.modo_auto and self.ultima_proba is not None:
            p = self.ultima_proba
            txt = self.fuente_chica.render(
                f"p(nada)={p[0]:.2f}  p(salto)={p[1]:.2f}  p(agachar)={p[2]:.2f}",
                True, self.AMARILLO,
            )
            self.pantalla.blit(txt, (10, y))
            y += self.fuente_chica.get_linesize() + 2

        # Indicador visual del estado actual
        if not self.modo_auto:
            estado_str = "AGACHADO" if self.agachado else ("SALTANDO" if self.salto else "suelo")
            tipo_str   = "bala ALTA (agacha)" if self.tipo_bala_actual == BALA_ALTA else "bala BAJA (salta)"
            color_tipo = (255, 120, 120) if self.tipo_bala_actual == BALA_ALTA else (120, 200, 255)
            s1 = self.fuente_chica.render(f"Estado: {estado_str}", True, self.GRIS)
            s2 = self.fuente_chica.render(tipo_str, True, color_tipo)
            self.pantalla.blit(s1, (10, y));  y += self.fuente_chica.get_linesize() + 2
            self.pantalla.blit(s2, (10, y))

    def loop(self) -> None:
        reloj = pygame.time.Clock()
        self.mostrar_menu()

        while self.corriendo:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    self.corriendo = False
                elif e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_q:
                        self.corriendo = False
                    elif e.key in (pygame.K_ESCAPE, pygame.K_p):
                        self._reset_estado_juego()
                        self.mostrar_menu()
                    elif e.key == pygame.K_f:
                        self._toggle_fullscreen()
                    elif not self.modo_auto:
                        # Saltar: ESPACIO
                        if e.key == pygame.K_SPACE and self.en_suelo and not self.agachado:
                            self.iniciar_salto()
                        # Agacharse: flecha abajo o S
                        elif e.key in (pygame.K_DOWN, pygame.K_s) and self.en_suelo and not self.salto:
                            self.iniciar_agache()

            if not self.corriendo:
                break

            # Lógica automática (IA)
            if self.modo_auto:
                accion = self.decision_auto()
                if accion == ACCION_SALTO:
                    self.iniciar_salto()
                elif accion == ACCION_AGACHAR:
                    self.iniciar_agache()
            else:
                # Registrar decisión manual para entrenamiento
                self.registrar_decision_manual()

            # Física
            if self.salto:
                self.manejar_salto()
            if self.agachado:
                self.manejar_agache()

            # Disparar bala si no hay una activa
            if not self.bala_disparada:
                self.disparar_bala()

            self._update_frame()
            pygame.display.flip()
            reloj.tick(45)

        pygame.quit()


def main() -> None:
    Juego().loop()


if __name__ == "__main__":
    main()