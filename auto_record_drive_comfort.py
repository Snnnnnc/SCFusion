#!/usr/bin/env python

# Copyright (c) 2024 CARLA Team.
# This work is licensed under the terms of the MIT license.

"""
Automatic driving with recording and comfort-based driving mode switching.
Based on auto_record_drive; adds:
- Wait for frontend "start" signal (poll --start-signal-url) before running simulation.
- Countdown (--countdown N seconds) after start for timestamp alignment with server.
- Receive comfort scores from server, time-weighted decision every 20s, switch driving
  style (cautious/normal/aggressive) by comfort level.
"""

from __future__ import print_function

import argparse
import collections
import datetime
import glob
import json
import logging
import math
import os
import numpy.random as random
import re
import sys
import time
import weakref

try:
    from urllib.request import urlopen, Request
    from urllib.error import URLError, HTTPError
except ImportError:
    urlopen = Request = None
    URLError = HTTPError = Exception

try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_q
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

# ==============================================================================
# -- Find CARLA module ---------------------------------------------------------
# ==============================================================================
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/carla')
except IndexError:
    pass

import carla
from carla import ColorConverter as cc

from agents.navigation.behavior_agent import BehaviorAgent
from agents.navigation.basic_agent import BasicAgent
from agents.navigation.constant_velocity_agent import ConstantVelocityAgent


# ==============================================================================
# -- Comfort-based driving mode switching --------------------------------------
# ==============================================================================

COMFORT_PREDICT_INTERVAL = 10.0   # 每 10s 接收一次舒适度评分
DECISION_INTERVAL = 20.0          # 每 20s 做一次是否切换驾驶模式的决策
LOOKBACK_WINDOW = 60.0            # 回看窗口 60s，决策时用窗口内评分做时间加权平均
TAU = 30.0                        # 时间常数 τ（秒），权重 W_i = exp(-(t0-ti)/τ)
RATE_LIMIT_PERIOD = 600.0         # 限频周期 10 分钟（秒）


def fetch_comfort_score(comfort_url, elapsed_time):
    """
    舒适度评分接收接口：从服务端拉取当前舒适度评分。
    每 10s 接收到的为 0-4 的整数，加权平均后得到 S_bar（可为小数），再按图二映射等级。
    :param comfort_url: 服务端 API URL（如 http://host:port/api/comfort），None 时返回随机 mock 值
    :param elapsed_time: 当前仿真相对开始时间（秒），用于 mock
    :return: 舒适度评分（0-4 整数，或服务端返回的数值）；请求失败时返回 None
    """
    if comfort_url and urlopen is not None:
        try:
            req = Request(comfort_url)
            resp = urlopen(req, timeout=2)
            data = json.loads(resp.read().decode())
            return float(data.get('comfort', data.get('score', 0)))
        except (URLError, HTTPError, ValueError, OSError) as e:
            logging.debug('Comfort fetch failed: %s', e)
            return None
    # Mock（无 URL 时）：随机返回 0-4 整数，加权后 S_bar 可能落在 Comfort/Mild/Bad，覆盖所有切换情况
    return float(random.randint(0, 5))  # 0,1,2,3,4  inclusive


def check_start_signal(start_signal_url):
    """
    轮询服务端是否已下发「开始」指令（前端点击开始后由服务端置位）。
    期望响应 JSON 含 "start": true 或 "status": "start"。
    """
    if not start_signal_url or urlopen is None:
        return False
    try:
        req = Request(start_signal_url)
        resp = urlopen(req, timeout=2)
        data = json.loads(resp.read().decode())
        if data.get('start') is True:
            return True, data
        if data.get('status') == 'start':
            return True, data
        return False, None
    except (URLError, HTTPError, ValueError, OSError):
        return False, None


def wait_for_start_signal(start_signal_url, world, display, clock, controller, args):
    """等待前端「开始」指令：保持渲染与 tick，轮询 start_signal_url，收到后返回。"""
    sync_fps = int(1.0 / 0.05) if args.sync else None
    world.hud.notification("Waiting for start signal...", seconds=999.0)
    while True:
        if sync_fps is not None:
            clock.tick(sync_fps)
        else:
            clock.tick()
        if args.sync:
            world.world.tick()
        else:
            world.world.wait_for_tick()
        if controller.parse_events():
            return False
        world.tick(clock)
        world.render(display)
        pygame.display.flip()
        ok, payload = check_start_signal(start_signal_url)
        if ok:
            return payload or {}
        time.sleep(0.3)


def run_countdown(seconds, world, display, clock, controller, args):
    """倒计时 seconds 秒（按真实时间），用于时间戳对齐（服务端与车端同时开始）。返回 False 若用户退出。"""
    sync_fps = int(1.0 / 0.05) if args.sync else None
    for n in range(seconds, 0, -1):
        world.hud.notification("Starting in %d..." % n, seconds=1.5)
        deadline = time.time() + 1.0
        while time.time() < deadline:
            if sync_fps is not None:
                clock.tick(sync_fps)
            else:
                clock.tick(20)
            if args.sync:
                world.world.tick()
            else:
                world.world.wait_for_tick()
            if controller.parse_events():
                return False
            world.tick(clock)
            world.render(display)
            pygame.display.flip()
    world.hud.notification("Go!", seconds=1.0)
    deadline = time.time() + 1.0
    while time.time() < deadline:
        if sync_fps is not None:
            clock.tick(sync_fps)
        else:
            clock.tick(20)
        if args.sync:
            world.world.tick()
        else:
            world.world.wait_for_tick()
        if controller.parse_events():
            return False
        world.tick(clock)
        world.render(display)
        pygame.display.flip()
    return True


def compute_weighted_comfort(buffer, t0):
    """
    决策时刻 t0 的时间加权平均舒适度 S(t0) = (Σ W_i * S_i) / (Σ W_i)，
    W_i = exp(-(t0 - t_i) / τ)，仅使用 [t0 - 60, t0] 内的评分。
    """
    if not buffer:
        return None
    t_min = t0 - LOOKBACK_WINDOW
    total_w = 0.0
    total_ws = 0.0
    for ti, si in buffer:
        if ti < t_min:
            continue
        w = math.exp(-(t0 - ti) / TAU)
        total_w += w
        total_ws += w * si
    if total_w <= 0:
        return None
    return total_ws / total_w


def comfort_level(s_bar):
    """加权评分 -> 舒适度等级：S_bar < 0.5 -> Comfort; 0.5 <= S_bar < 1.5 -> Mild; S_bar >= 1.5 -> Bad"""
    if s_bar is None:
        return None
    if s_bar < 0.5:
        return 'Comfort'
    if s_bar < 1.5:
        return 'Mild'
    return 'Bad'


def next_behavior_for_level(level, current_behavior):
    """
    根据舒适度等级与当前驾驶模式，得到应切换到的模式（可能不变）。
    Comfort: 不切换；Mild: 回退一级；Bad: 直接 cautious。
    """
    order = ['cautious', 'normal', 'aggressive']
    if level == 'Comfort':
        return current_behavior
    if level == 'Mild':
        idx = order.index(current_behavior) if current_behavior in order else 1
        return order[max(0, idx - 1)]
    if level == 'Bad':
        return 'cautious'
    return current_behavior


# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================

def find_weather_presets():
    """Method to find weather presets"""
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    def name(x): return ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    """Method to get actor display name"""
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name


def get_actor_blueprints(world, filter, generation):
    bps = world.get_blueprint_library().filter(filter)

    if generation.lower() == "all":
        return bps

    if len(bps) == 1:
        return bps

    try:
        int_generation = int(generation)
        if int_generation in [1, 2, 3]:
            bps = [x for x in bps if int(x.get_attribute('generation')) == int_generation]
            return bps
        else:
            print("   Warning! Actor Generation is not valid. No actor will be spawned.")
            return []
    except:
        print("   Warning! Actor Generation is not valid. No actor will be spawned.")
        return []


def list_available_options(client):
    """Display available vehicle models, maps, and spawn points"""
    try:
        world = client.get_world()
        
        print("\n" + "="*60)
        print("Available Options:")
        print("="*60)
        
        # Display available maps
        print("\nAvailable Maps:")
        available_maps = client.get_available_maps()
        for i, map_name in enumerate(available_maps):
            map_short_name = map_name.split('/')[-1]
            print(f"  {i}: {map_short_name}")
        
        # Display available vehicle models
        print("\nAvailable Vehicle Models:")
        blueprint_library = world.get_blueprint_library()
        vehicle_bps = blueprint_library.filter('vehicle.*')
        for i, bp in enumerate(vehicle_bps):
            print(f"  {i}: {bp.id}")
        
        # Display spawn points
        print(f"\nNumber of Spawn Points: {len(world.get_map().get_spawn_points())}")
        print("Use --spawn-point and --destination-point to specify indices (0-based)")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"Error listing options: {e}")


def override_speed_limits(sim_world, new_limit=None):
    """将地图中的道路限速整体调整为 new_limit（km/h）"""
    if new_limit is None:
        return

    try:
        speed_limit_actors = sim_world.get_actors().filter('traffic.speed_limit.*')
    except RuntimeError as e:
        print(f"[SpeedLimit] 获取限速标记失败: {e}")
        return

    changed = 0
    for actor in speed_limit_actors:
        try:
            if hasattr(actor, "set_speed_limit"):
                actor.set_speed_limit(float(new_limit))
            else:
                # 兼容旧版：部分 TrafficSign 没有 setter，只能直接修改属性
                if hasattr(actor, "speed_limit"):
                    actor.speed_limit = float(new_limit)
                else:
                    continue
            changed += 1
        except RuntimeError as e:
            print(f"[SpeedLimit] 设置限速失败: {e}")

    if changed:
        print(f"[SpeedLimit] 已将 {changed} 个道路限速标记统一设置为 {new_limit} km/h")
    else:
        print("[SpeedLimit] 未找到可设置的限速标记")


# ==============================================================================
# -- World ---------------------------------------------------------------------
# ==============================================================================

class World(object):
    """Class representing the surrounding environment"""

    def __init__(self, carla_world, hud, args):
        """Constructor method"""
        self._args = args
        self.world = carla_world
        try:
            self.map = self.world.get_map()
        except RuntimeError as error:
            print('RuntimeError: {}'.format(error))
            print('  The server could not send the OpenDRIVE (.xodr) file:')
            print('  Make sure it exists, has the same name of your town, and is correct.')
            sys.exit(1)
        self.hud = hud
        self.player = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.camera_manager = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._actor_filter = args.filter
        self._actor_generation = args.generation
        self.restart(args)
        self.world.on_tick(hud.on_world_tick)
        self.recording_enabled = False
        self.recording_start = 0

    def restart(self, args):
        """Restart the world"""
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_id = self.camera_manager.transform_index if self.camera_manager is not None else 0

        # Get blueprint - support specific vehicle model
        if args.vehicle_model:
            blueprint_library = self.world.get_blueprint_library()
            try:
                blueprint = blueprint_library.find(args.vehicle_model)
                print(f"Using specified vehicle model: {args.vehicle_model}")
            except:
                print(f"Vehicle model {args.vehicle_model} not found, using filter")
                blueprint_list = get_actor_blueprints(self.world, self._actor_filter, self._actor_generation)
                if not blueprint_list:
                    raise ValueError("Couldn't find any blueprints with the specified filters")
                blueprint = random.choice(blueprint_list)
        else:
            blueprint_list = get_actor_blueprints(self.world, self._actor_filter, self._actor_generation)
            if not blueprint_list:
                raise ValueError("Couldn't find any blueprints with the specified filters")
            blueprint = random.choice(blueprint_list)
        
        blueprint.set_attribute('role_name', 'hero')
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)

        # Spawn the player
        if self.player is not None:
            spawn_point = self.player.get_transform()
            spawn_point.location.z += 2.0
            spawn_point.rotation.roll = 0.0
            spawn_point.rotation.pitch = 0.0
            self.destroy()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
            self.modify_vehicle_physics(self.player)
        
        while self.player is None:
            if not self.map.get_spawn_points():
                print('There are no spawn points available in your map/town.')
                print('Please add some Vehicle Spawn Point to your UE4 scene.')
                sys.exit(1)
            spawn_points = self.map.get_spawn_points()
            
            # Support fixed spawn point
            if args.spawn_point is not None and 0 <= args.spawn_point < len(spawn_points):
                spawn_point = spawn_points[args.spawn_point]
                print(f"Using fixed spawn point {args.spawn_point}: ({spawn_point.location.x:.1f}, {spawn_point.location.y:.1f})")
            else:
                spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
            
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
            self.modify_vehicle_physics(self.player)

        if self._args.sync:
            self.world.tick()
        else:
            self.world.wait_for_tick()

        # Set up the sensors
        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.gnss_sensor = GnssSensor(self.player)
        self.camera_manager = CameraManager(self.player, self.hud)
        self.camera_manager.transform_index = cam_pos_id
        self.camera_manager.set_sensor(cam_index, notify=False)
        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)

    def next_weather(self, reverse=False):
        """Get next weather setting"""
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.hud.notification('Weather: %s' % preset[1])
        self.player.get_world().set_weather(preset[0])

    def modify_vehicle_physics(self, actor):
        try:
            physics_control = actor.get_physics_control()
            physics_control.use_sweep_wheel_collision = True
            actor.apply_physics_control(physics_control)
        except Exception:
            pass

    def tick(self, clock):
        """Method for every tick"""
        self.hud.tick(self, clock)

    def render(self, display):
        """Render world"""
        self.camera_manager.render(display)
        self.hud.render(display)

    def destroy_sensors(self):
        """Destroy sensors"""
        self.camera_manager.sensor.destroy()
        self.camera_manager.sensor = None
        self.camera_manager.index = None

    def destroy(self):
        """Destroys all actors"""
        actors = [
            self.camera_manager.sensor,
            self.collision_sensor.sensor,
            self.lane_invasion_sensor.sensor,
            self.gnss_sensor.sensor,
            self.player]
        for actor in actors:
            if actor is not None:
                actor.destroy()


# ==============================================================================
# -- KeyboardControl -----------------------------------------------------------
# ==============================================================================

class KeyboardControl(object):
    def __init__(self, world):
        world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)

    def parse_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            if event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True

    @staticmethod
    def _is_quit_shortcut(key):
        """Shortcut for quitting"""
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)


# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================

class HUD(object):
    """Class for HUD text"""

    def __init__(self, width, height):
        """Constructor method"""
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.help = HelpText(pygame.font.Font(mono, 24), width, height)
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()

    def on_world_tick(self, timestamp):
        """Gets informations from the world at every tick"""
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame_count
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, clock):
        """HUD method for every tick"""
        self._notifications.tick(world, clock)
        if not self._show_info:
            return
        transform = world.player.get_transform()
        vel = world.player.get_velocity()
        control = world.player.get_control()
        heading = 'N' if abs(transform.rotation.yaw) < 89.5 else ''
        heading += 'S' if abs(transform.rotation.yaw) > 90.5 else ''
        heading += 'E' if 179.5 > transform.rotation.yaw > 0.5 else ''
        heading += 'W' if -0.5 > transform.rotation.yaw > -179.5 else ''
        colhist = world.collision_sensor.get_collision_history()
        collision = [colhist[x + self.frame - 200] for x in range(0, 200)]
        max_col = max(1.0, max(collision))
        collision = [x / max_col for x in collision]
        vehicles = world.world.get_actors().filter('vehicle.*')

        self._info_text = [
            'Server:  % 16.0f FPS' % self.server_fps,
            'Client:  % 16.0f FPS' % clock.get_fps(),
            '',
            'Vehicle: % 20s' % get_actor_display_name(world.player, truncate=20),
            'Map:     % 20s' % world.map.name.split('/')[-1],
            'Simulation time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)),
            '',
            'Speed:   % 15.0f km/h' % (3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)),
        ]
        if getattr(world, 'current_driving_behavior', None) is not None:
            self._info_text += ['Mode:    % 20s' % world.current_driving_behavior, '']
        self._info_text += [
            u'Heading:% 16.0f\N{DEGREE SIGN} % 2s' % (transform.rotation.yaw, heading),
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (transform.location.x, transform.location.y)),
            'GNSS:% 24s' % ('(% 2.6f, % 3.6f)' % (world.gnss_sensor.lat, world.gnss_sensor.lon)),
            'Height:  % 18.0f m' % transform.location.z,
            '']
        if isinstance(control, carla.VehicleControl):
            self._info_text += [
                ('Throttle:', control.throttle, 0.0, 1.0),
                ('Steer:', control.steer, -1.0, 1.0),
                ('Brake:', control.brake, 0.0, 1.0),
                ('Reverse:', control.reverse),
                ('Hand brake:', control.hand_brake),
                ('Manual:', control.manual_gear_shift),
                'Gear:        %s' % {-1: 'R', 0: 'N'}.get(control.gear, control.gear)]
        elif isinstance(control, carla.WalkerControl):
            self._info_text += [
                ('Speed:', control.speed, 0.0, 5.556),
                ('Jump:', control.jump)]
        self._info_text += [
            '',
            'Collision:',
            collision,
            '',
            'Number of vehicles: % 8d' % len(vehicles)]

        if len(vehicles) > 1:
            self._info_text += ['Nearby vehicles:']

        def dist(l):
            return math.sqrt((l.x - transform.location.x)**2 + (l.y - transform.location.y)**2 + (l.z - transform.location.z)**2)
        vehicles = [(dist(x.get_location()), x) for x in vehicles if x.id != world.player.id]

        for dist, vehicle in sorted(vehicles):
            if dist > 200.0:
                break
            vehicle_type = get_actor_display_name(vehicle, truncate=22)
            self._info_text.append('% 4dm %s' % (dist, vehicle_type))

    def toggle_info(self):
        """Toggle info on or off"""
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        """Notification text"""
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        """Error text"""
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        """Render for HUD class"""
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1 - y) * 30) for x, y in enumerate(item)]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        fig = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect((bar_h_offset + fig * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect((bar_h_offset, v_offset + 8), (fig * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
        self._notifications.render(display)
        self.help.render(display)


# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================

class FadingText(object):
    """Class for fading text"""

    def __init__(self, font, dim, pos):
        """Constructor method"""
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        """Set fading text"""
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        """Fading text method for every tick"""
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        """Render fading text method"""
        display.blit(self.surface, self.pos)


# ==============================================================================
# -- HelpText ------------------------------------------------------------------
# ==============================================================================

class HelpText(object):
    """Helper class for text render"""

    def __init__(self, font, width, height):
        """Constructor method"""
        lines = __doc__.split('\n')
        self.font = font
        self.dim = (680, len(lines) * 22 + 12)
        self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for i, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, i * 22))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        """Toggle on or off the render help"""
        self._render = not self._render

    def render(self, display):
        """Render help text method"""
        if self._render:
            display.blit(self.surface, self.pos)


# ==============================================================================
# -- CollisionSensor -----------------------------------------------------------
# ==============================================================================

class CollisionSensor(object):
    """Class for collision sensors"""

    def __init__(self, parent_actor, hud):
        """Constructor method"""
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        blueprint = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(blueprint, carla.Transform(), attach_to=self._parent)
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        """Gets the history of collisions"""
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        """On collision method"""
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        self.hud.notification('Collision with %r' % actor_type)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)


# ==============================================================================
# -- LaneInvasionSensor --------------------------------------------------------
# ==============================================================================

class LaneInvasionSensor(object):
    """Class for lane invasion sensors"""

    def __init__(self, parent_actor, hud):
        """Constructor method"""
        self.sensor = None
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        """On invasion method"""
        self = weak_self()
        if not self:
            return
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split()[-1] for x in lane_types]
        self.hud.notification('Crossed line %s' % ' and '.join(text))


# ==============================================================================
# -- GnssSensor ----------------------------------------------------------------
# ==============================================================================

class GnssSensor(object):
    """Class for GNSS sensors"""

    def __init__(self, parent_actor):
        """Constructor method"""
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        world = self._parent.get_world()
        blueprint = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(blueprint, carla.Transform(carla.Location(x=1.0, z=2.8)),
                                        attach_to=self._parent)
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        """GNSS method"""
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude


# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================

class CameraManager(object):
    """Class for camera management"""

    def __init__(self, parent_actor, hud):
        """Constructor method"""
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = False
        bound_x = 0.5 + self._parent.bounding_box.extent.x
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        bound_z = 0.5 + self._parent.bounding_box.extent.z
        attachment = carla.AttachmentType
        self._camera_transforms = [
            (carla.Transform(carla.Location(x=-2.0*bound_x, y=+0.0*bound_y, z=2.0*bound_z), carla.Rotation(pitch=8.0)), attachment.SpringArmGhost),
            (carla.Transform(carla.Location(x=+0.8*bound_x, y=+0.0*bound_y, z=1.3*bound_z)), attachment.Rigid),
            (carla.Transform(carla.Location(x=+1.9*bound_x, y=+1.0*bound_y, z=1.2*bound_z)), attachment.SpringArmGhost),
            (carla.Transform(carla.Location(x=-2.8*bound_x, y=+0.0*bound_y, z=4.6*bound_z), carla.Rotation(pitch=6.0)), attachment.SpringArmGhost),
            (carla.Transform(carla.Location(x=-1.0, y=-1.0*bound_y, z=0.4*bound_z)), attachment.Rigid)]

        self.transform_index = 1
        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB'],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)'],
            ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)'],
            ['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)'],
            ['sensor.camera.semantic_segmentation', cc.Raw, 'Camera Semantic Segmentation (Raw)'],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette,
             'Camera Semantic Segmentation (CityScapes Palette)'],
            ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)']]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            blp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                blp.set_attribute('image_size_x', str(hud.dim[0]))
                blp.set_attribute('image_size_y', str(hud.dim[1]))
            elif item[0].startswith('sensor.lidar'):
                blp.set_attribute('range', '50')
            item.append(blp)
        self.index = None

    def toggle_camera(self):
        """Activate a camera"""
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        self.set_sensor(self.index, notify=False, force_respawn=True)

    def set_sensor(self, index, notify=True, force_respawn=False):
        """Set a sensor"""
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None else (
            force_respawn or (self.sensors[index][0] != self.sensors[self.index][0]))
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self._camera_transforms[self.transform_index][0],
                attach_to=self._parent,
                attachment_type=self._camera_transforms[self.transform_index][1])

            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    def next_sensor(self):
        """Get the next sensor"""
        self.set_sensor(self.index + 1)

    def toggle_recording(self):
        """Toggle recording on or off"""
        self.recording = not self.recording
        self.hud.notification('Recording %s' % ('On' if self.recording else 'Off'))

    def render(self, display):
        """Render method"""
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        if self.sensors[self.index][0].startswith('sensor.lidar'):
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 4), 4))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self.hud.dim) / 100.0
            lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            lidar_data = np.fabs(lidar_data)
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
            lidar_img = np.zeros(lidar_img_size)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self.surface = pygame.surfarray.make_surface(lidar_img)
        else:
            image.convert(self.sensors[self.index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if self.recording:
            image.save_to_disk('_out/%08d' % image.frame)


# ==============================================================================
# -- Game Loop -----------------------------------------------------------------
# ==============================================================================

def game_loop(args):
    """
    Main loop of the simulation. It handles updating all the HUD information,
    ticking the agent and, if needed, the world.
    """

    pygame.init()
    pygame.font.init()
    world = None
    recording_filename = None

    try:
        if args.seed:
            random.seed(args.seed)

        client = carla.Client(args.host, args.port)
        client.set_timeout(60.0)

        # Load specific map if requested
        if args.map:
            print(f"Loading map: {args.map}")
            try:
                sim_world = client.load_world(args.map)
                print(f"Successfully loaded map: {args.map}")
            except Exception as e:
                print(f"Failed to load map {args.map}: {e}")
                print("Available maps:")
                available_maps = client.get_available_maps()
                for map_name in available_maps:
                    print(f"  - {map_name}")
                return
        else:
            sim_world = client.get_world()

        traffic_manager = client.get_trafficmanager()

        if args.sync:
            settings = sim_world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05
            sim_world.apply_settings(settings)
            traffic_manager.set_synchronous_mode(True)

        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

        hud = HUD(args.width, args.height)
        world = World(client.get_world(), hud, args)
        controller = KeyboardControl(world)
        
        # Setup agent
        if args.speed_limit_override is not None:
            override_speed_limits(sim_world, args.speed_limit_override)

        if args.agent == "Basic":
            agent = BasicAgent(world.player, 30)
            agent.follow_speed_limits(True)
        elif args.agent == "Constant":
            agent = ConstantVelocityAgent(world.player, 30)
            ground_loc = world.world.ground_projection(world.player.get_location(), 5)
            if ground_loc:
                world.player.set_location(ground_loc.location + carla.Location(z=0.01))
            agent.follow_speed_limits(True)
        elif args.agent == "Behavior":
            agent = BehaviorAgent(world.player, behavior=args.behavior)
           
            agent.get_local_planner().follow_speed_limits(False)
            agent.set_target_speed(90)
            agent.max_throttle = 0.9
            agent.brake_threshold = 0.2
            agent.overtake_allowed = True

        # Set the agent destination
        spawn_points = world.map.get_spawn_points()
        
        if args.destination_point is not None and 0 <= args.destination_point < len(spawn_points):
            destination = spawn_points[args.destination_point].location
            print(f"Using fixed destination point {args.destination_point}: ({destination.x:.1f}, {destination.y:.1f})")
        else:
            destination = random.choice(spawn_points).location
        
        agent.set_destination(destination)

        # Comfort-based mode switching state (only when Behavior agent and comfort enabled)
        comfort_enabled = (args.agent == "Behavior" and not getattr(args, 'no_comfort', False))
        comfort_buffer = []           # list of (t, s) in simulation seconds since start
        last_predict_time = -COMFORT_PREDICT_INTERVAL - 1
        last_decision_time = -DECISION_INTERVAL - 1
        last_switch_time = -RATE_LIMIT_PERIOD - 1
        last_decision_level = None    # 'Comfort' | 'Mild' | 'Bad'
        current_behavior = args.behavior

        # Generate recording filename（录制在收到开始指令并倒计时后再启动）
        map_name = world.map.name.split('/')[-1]
        vehicle_name = world.player.type_id.replace('.', '_')
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if args.recorder_filename:
            recording_filename = args.recorder_filename
        else:
            recording_filename = f"recording_{vehicle_name}_{map_name}_{timestamp}.log"

        simulation_time_limit = args.simulation_time  # 固定仿真时间（秒）
        clock = pygame.time.Clock()
        sync_target_fps = int(1.0 / 0.05) if args.sync else None  # 20

        # 等待前端「开始」指令，再倒计时，保证与服务端时间戳对齐
        start_signal_url = getattr(args, 'start_signal_url', None)
        if not start_signal_url and getattr(args, 'comfort_url', None):
            base = args.comfort_url.rstrip('/')
            if '/api/' in base:
                start_signal_url = base.rsplit('/api/', 1)[0] + '/api/start'
            else:
                start_signal_url = base + '/api/start'
        start_payload = {}
        if start_signal_url:
            print("Waiting for start signal from: %s" % start_signal_url)
            start_payload = wait_for_start_signal(start_signal_url, world, display, clock, controller, args)
            if start_payload is False:
                return

            # 优先使用服务端下发的 start_at（epoch seconds），要求两台机器时钟同步（NTP）
            start_at = start_payload.get("start_at", None) if isinstance(start_payload, dict) else None
            if start_at is not None:
                try:
                    start_at = float(start_at)
                    countdown_sec = max(0, int(math.ceil(start_at - time.time())))
                except Exception:
                    countdown_sec = max(0, getattr(args, 'countdown', 5))
            else:
                countdown_sec = max(0, getattr(args, 'countdown', 5))

            if countdown_sec > 0:
                print("Countdown %d s for timestamp alignment..." % countdown_sec)
                if not run_countdown(countdown_sec, world, display, clock, controller, args):
                    return

        # 倒计时结束后再开始录制与仿真
        print(f"Starting recording: {recording_filename}")
        client.start_recorder(recording_filename)
        world.recording_enabled = True
        world.hud.notification(f"Recording to {recording_filename}", seconds=5.0)

        start_time = None

        while True:
            if sync_target_fps is not None:
                clock.tick(sync_target_fps)
            else:
                clock.tick()
            if args.sync:
                world.world.tick()
            else:
                world.world.wait_for_tick()
            
            # 记录开始时间
            if start_time is None:
                snapshot = world.world.get_snapshot()
                start_time = snapshot.timestamp.elapsed_seconds
            
            # 检查是否超过固定仿真时间
            if simulation_time_limit is not None:
                snapshot = world.world.get_snapshot()
                elapsed_time = snapshot.timestamp.elapsed_seconds - start_time
                
                if elapsed_time >= simulation_time_limit:
                    print(f"\nSimulation time limit reached: {elapsed_time:.1f}s / {simulation_time_limit:.1f}s")
                    break
            else:
                snapshot = world.world.get_snapshot()
                elapsed_time = snapshot.timestamp.elapsed_seconds - start_time

            # 舒适度：每 10s 接收一次评分；每 20s 做一次是否切换驾驶模式的决策
            if comfort_enabled and elapsed_time >= 0:
                # 每 10s 拉取一次舒适度评分
                if elapsed_time - last_predict_time >= COMFORT_PREDICT_INTERVAL:
                    score = fetch_comfort_score(getattr(args, 'comfort_url', None), elapsed_time)
                    if score is not None:
                        comfort_buffer.append((elapsed_time, score))
                        # 只保留 60s 内
                        comfort_buffer = [(t, s) for t, s in comfort_buffer if t >= elapsed_time - LOOKBACK_WINDOW]
                    last_predict_time = elapsed_time

                # 每 20s 做一次决策
                if elapsed_time - last_decision_time >= DECISION_INTERVAL:
                    s_bar = compute_weighted_comfort(comfort_buffer, elapsed_time)
                    level = comfort_level(s_bar)
                    last_decision_time = elapsed_time

                    if level is not None:
                        # 连续两次 Bad 则忽略限频，强制切到 cautious
                        two_consecutive_bad = (last_decision_level == 'Bad' and level == 'Bad')
                        within_rate_limit = (elapsed_time - last_switch_time) < RATE_LIMIT_PERIOD

                        if two_consecutive_bad:
                            new_behavior = 'cautious'
                            do_switch = True
                        elif within_rate_limit and not two_consecutive_bad:
                            do_switch = False
                            new_behavior = current_behavior
                        else:
                            new_behavior = next_behavior_for_level(level, current_behavior)
                            do_switch = (new_behavior != current_behavior)

                        if do_switch and new_behavior != current_behavior:
                            agent.set_behavior(new_behavior)
                            current_behavior = new_behavior
                            last_switch_time = elapsed_time
                            world.hud.notification("Mode: %s" % new_behavior, seconds=3.0)
                            logging.info("Comfort level %s -> behavior %s (S_bar=%.3f)", level, new_behavior, s_bar or 0)

                        last_decision_level = level

            if args.agent == "Behavior":
                world.current_driving_behavior = current_behavior

            if controller.parse_events():
                return

            world.tick(clock)
            world.render(display)
            pygame.display.flip()

            if agent.done():
                # 如果设置了固定仿真时间
                if simulation_time_limit is not None:
                    snapshot = world.world.get_snapshot()
                    elapsed_time = snapshot.timestamp.elapsed_seconds - start_time
                    remaining_time = simulation_time_limit - elapsed_time

                    if remaining_time > 0:
                        # 还有剩余时间，设置新目标继续
                        if args.destination_point is not None and 0 <= args.destination_point < len(spawn_points):
                            destination = spawn_points[args.destination_point].location
                        else:
                            destination = random.choice(spawn_points).location
                        
                        agent.set_destination(destination)
                        world.hud.notification(f"Target reached! New destination set. Remaining time: {remaining_time:.1f}s", seconds=4.0)
                        print(f"Target reached at {elapsed_time:.1f}s. Setting new destination. Remaining: {remaining_time:.1f}s")
                    else:
                        # 时间已到
                        print("Target reached and simulation time limit reached")
                        break
                
                else:
                    if args.loop:
                        if args.destination_point is not None and 0 <= args.destination_point < len(spawn_points):
                            destination = spawn_points[args.destination_point].location
                        else:
                            destination = random.choice(spawn_points).location
                        agent.set_destination(destination)
                        world.hud.notification("Target reached", seconds=4.0)
                        print("Destination: ", destination)
                        print("The target has been reached, searching for another target")
                    else:
                        print("Destination: ", destination)
                        print("The target has been reached, stopping the simulation")
                        break

            control = agent.run_step()
            control.manual_gear_shift = False
            world.player.apply_control(control)

    finally:
        # Stop recording
        if world is not None and world.recording_enabled:
            client.stop_recorder()
            print(f"\nRecording saved to: {recording_filename}")
        
        if world is not None:
            settings = world.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.world.apply_settings(settings)
            traffic_manager.set_synchronous_mode(True)

            world.destroy()

        pygame.quit()


# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================

def main():
    """Main method"""

    argparser = argparse.ArgumentParser(
        description='CARLA Automatic Control with Recording')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='Print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='Window resolution (default: 1280x720)')
    argparser.add_argument(
        '--sync',
        action='store_true',
        help='Synchronous mode execution')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.*',
        help='Actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '--generation',
        metavar='G',
        default='2',
        help='restrict to certain actor generation (values: "1","2","All" - default: "2")')
    argparser.add_argument(
        '-l', '--loop',
        action='store_true',
        dest='loop',
        help='Sets a new random destination upon reaching the previous one (default: False)')
    argparser.add_argument(
        "-a", "--agent", type=str,
        choices=["Behavior", "Basic", "Constant"],
        help="select which agent to run",
        default="Behavior")
    argparser.add_argument(
        '-b', '--behavior', type=str,
        choices=["cautious", "normal", "aggressive"],
        help='Choose one of the possible agent behaviors (default: aggressive) ',
        default='aggressive')
    argparser.add_argument(
        '-s', '--seed',
        help='Set seed for repeating executions (default: None)',
        default=None,
        type=int)
    argparser.add_argument(
        '--map',
        metavar='TOWN',
        default=None,
        help='Load specific map (e.g., Town01, Town02, etc.)')
    argparser.add_argument(
        '--speed-limit-override',
        type=float,
        default=None,
        help='统一修改地图中所有道路限速（单位 km/h），默认不修改')
    argparser.add_argument(
        '--vehicle-model',
        metavar='MODEL',
        default='vehicle.tesla.model3',
        help='Specific vehicle model (e.g., vehicle.tesla.model3)')
    argparser.add_argument(
        '--spawn-point',
        metavar='INDEX',
        default=None,
        type=int,
        help='Fixed spawn point index (default: random)')
    argparser.add_argument(
        '--destination-point',
        metavar='INDEX',
        default=None,
        type=int,
        help='Fixed destination point index (default: random)')
    argparser.add_argument(
        '--simulation-time',
        metavar='SECONDS',
        default=360.0,
        type=float,
        help='Fixed simulation time in seconds'
    )
    argparser.add_argument(
        '--auto-continue',
        action='store_true',
        help='Auto continue the simulation after the destination is reached'
    )
    argparser.add_argument(
        '-f', '--recorder-filename',
        metavar='F',
        default=None,
        help='Custom recorder filename (default: auto-generated)')
    argparser.add_argument(
        '--list-options',
        action='store_true',
        help='List available maps, vehicles, and spawn points')
    argparser.add_argument(
        '--comfort-url',
        metavar='URL',
        default=None,
        help='Comfort score API URL (e.g. http://127.0.0.1:5000/api/comfort). If not set, use mock score.')
    argparser.add_argument(
        '--no-comfort',
        action='store_true',
        dest='no_comfort',
        help='Disable comfort-based driving mode switching')
    argparser.add_argument(
        '--start-signal-url',
        metavar='URL',
        default=None,
        help='URL to poll for start signal (e.g. http://127.0.0.1:5000/api/start). If not set, derived from --comfort-url.')
    argparser.add_argument(
        '--countdown',
        type=int,
        metavar='SECONDS',
        default=5,
        help='Countdown seconds after start signal for timestamp alignment (default: 5)')

    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    # List options if requested
    if args.list_options:
        client = carla.Client(args.host, args.port)
        client.set_timeout(10.0)
        list_available_options(client)
        return

    print(__doc__)

    try:
        game_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':
    main()

