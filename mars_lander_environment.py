""" 
Rocket trajectory optimization is a classic topic in Optimal Control.

According to Pontryagin's maximum principle it's optimal to fire engine full throttle or
turn it off. That's the reason this environment is OK to have discreet actions (engine on or off).

The landing pad is always at coordinates (0,0). 

State Vector
------------

The coordinates are the first two numbers in the state vector.
        state = [
            # SkyCrane Data
            x,              #0  -100%...100%
            y,              #1 -100%...100%
            velocity.x,     #2
            velocity.y, #   3
            skycrane.angle, #4
            angularVelocity, #5
            # Lander Data
            legs[0].ground_contact , #6
            legs[1].ground_contact,  #7
            pos_lander.x ,  #8
            pos_lander.y ,  #9
            vel_lander.x,   #10
            vel_lander.y,   #11
            lander.angle,   #12
            lander.angularVelocity, #13
            tether_connected,       #14
        ]

Action
------
OUTSIDE ENGINES BRANCH - The mars sky crane had downward directed side thrusters only

0 No operation
1 Fire left engine full
2 Fire both engines half
3 Fire right engine full
4 release tether


release_tether is dependent on tether_action=True in MarsLander.__init__()


Reward for moving from the top of the screen to the landing pad and zero speed is about 100..140 points.
If the lander moves away from the landing pad it loses reward. The episode finishes if the lander crashes or
comes to rest, receiving an additional -100 or +100 points. Each leg with ground contact is +10 points.
Firing the main engine is -0.3 points each frame. Firing the side engine is -0.03 points each frame.
Solved is 200 points.

Landing outside the landing pad is possible. Fuel is infinite, so an agent can learn to fly and then land
on its first attempt. Please see the source code for details.



BASED ON
LunarLander-v2
Created by Oleg Klimov. Licensed on the same terms as the rest of OpenAI Gym.
Copied from openai.gym

Changelog
Additional GAME OVER conditions - max angle of bodies 90 DEG either way




"""


import math
import numpy as np

import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, # type: ignore
    polygonShape, revoluteJointDef, contactListener, distanceJointDef, ropeJointDef) # type: ignore


import gym
from gym import spaces
from gym.utils import seeding, EzPickle

FPS = 50
SCALE = 30.0   # affects how fast-paced the game is, forces should be adjusted as well

FULL_ENGINE_POWER = 26.0 # single engine thrust
HALF_ENGINE_POWER = 26.0   # both engines, each. not really linear here

INITIAL_RANDOM = 700.0   # Set 1500 to make game harder

LANDER_POLY =[
    (-14, +17), (-17, 0), (-17 ,-10),
    (+17, -10), (+17, 0), (+14, +17)
    ]
LEG_AWAY = 20
LEG_DOWN = 18
LEG_W, LEG_H = 2, 8
LEG_SPRING_TORQUE = 50

TETHER_LENGTH = 64
SKYCRANE_POLY =[
    (-24, +10), (-34, 0), (-34 ,-10),
    (+34, -10), (+34, 0), (+24, +10)
    ]

SIDE_ENGINE_HEIGHT = 10.0 # POSITIVE IS DOWN???
SIDE_ENGINE_AWAY = 34.0
SIDE_ENGINE_ANGLE = 15*(math.pi*2/360) # from y-Axis outwards

VIEWPORT_W = 600
VIEWPORT_H = 400


class ContactDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        # if the lander has contact to anything, it has crashed
        if self.env.lander in [contact.fixtureA.body, contact.fixtureB.body]:
            self.env.game_over = True
        
        # if the SkyCrane has contact to anything, it also has crashed
        if self.env.skycrane in [contact.fixtureA.body, contact.fixtureB.body]:
            self.env.game_over = True        

        # if the legs have contact to something, set the ground contact flags
        for i in range(2):
            if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.legs[i].ground_contact = True

    def EndContact(self, contact):
        for i in range(2):
            if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.legs[i].ground_contact = False


class MarsLander(gym.Env, EzPickle):

    # TODO : Define a wind force.
    # e.g.
    # https://gamedev.stackexchange.com/questions/158653/simulate-wind-affecting-a-boat-using-a-2d-physics-engine-like-box2d-or-spritekit
    # workaround is sideways gravity vector


    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : FPS
    }

    actions_dict = {
        0: 'No operation',
        1: 'Fire left engine full',
        2: 'Fire both engines half',
        3: 'Fire right engine full',
        4: 'release tether',
    }

    def __init__(self, gravitympss:float=3.721, tether_action:bool=False, render_reward_indicator:bool=False):
        EzPickle.__init__(self)
        self.seed()
        self.viewer = None

        self.world = Box2D.b2World(gravity=(0, -gravitympss))
        self.moon = None
        self.lander = None
        self.skycrane = None
        self.particles = []

        self.prev_reward = None
        self.render_reward_indicator = render_reward_indicator
        # useful range is -1 .. +1, but spikes can be higher
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(15,), dtype=np.float32)

        if tether_action:
            self.action_space = spaces.Discrete(5)
        else:
            self.action_space = spaces.Discrete(4)

        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _destroy(self):
        if not self.moon: return
        self.world.contactListener = None
        self._clean_particles(True)
        self.world.DestroyBody(self.moon)
        self.moon = None
        self.world.DestroyBody(self.lander)
        self.lander = None
        self.world.DestroyBody(self.skycrane)
        self.skycrane = None
        self.world.DestroyBody(self.legs[0])
        self.world.DestroyBody(self.legs[1])

    def reset(self):
        self._destroy()
        self.world.contactListener_keepref = ContactDetector(self) #type:ignore
        self.world.contactListener = self.world.contactListener_keepref #type:ignore
        self.game_over = False
        self.prev_shaping = None

        W = VIEWPORT_W/SCALE
        H = VIEWPORT_H/SCALE
        ###########################
        # The Mars (called Moon in the code :-)
        # terrain
        ###########################
        CHUNKS = 11
        height = self.np_random.uniform(0, H/2, size=(CHUNKS+1,))
        chunk_x = [W/(CHUNKS-1)*i for i in range(CHUNKS)]
        self.helipad_x1 = chunk_x[CHUNKS//2-1]
        self.helipad_x2 = chunk_x[CHUNKS//2+1]
        self.helipad_y = H/4
        height[CHUNKS//2-2] = self.helipad_y
        height[CHUNKS//2-1] = self.helipad_y
        height[CHUNKS//2+0] = self.helipad_y
        height[CHUNKS//2+1] = self.helipad_y
        height[CHUNKS//2+2] = self.helipad_y
        smooth_y = [0.33*(height[i-1] + height[i+0] + height[i+1]) for i in range(CHUNKS)]

        self.moon = self.world.CreateStaticBody(shapes=edgeShape(vertices=[(0, 0), (W, 0)]))
        self.sky_polys = []
        for i in range(CHUNKS-1):
            p1 = (chunk_x[i], smooth_y[i])
            p2 = (chunk_x[i+1], smooth_y[i+1])
            self.moon.CreateEdgeFixture(
                vertices=[p1,p2],
                density=0,
                friction=0.5)
            self.sky_polys.append([p1, p2, (p2[0], H), (p1[0], H)])

        self.moon.color1 = (0.5, 0.0, 0.0)
        self.moon.color2 = (0.5, 0.0, 0.0)


        ########################
        # The SkyCrane
        #
        # The SkyCrane that carries the lander on a tether (Box2D: distance joint)
        ########################

        initial_y = VIEWPORT_H/SCALE
        self.skycrane = self.world.CreateDynamicBody(
            position=(VIEWPORT_W/SCALE/2, initial_y),
            angle=0.0,
            fixtures = fixtureDef(
                shape=polygonShape(vertices=[(x/SCALE, y/SCALE) for x, y in SKYCRANE_POLY]),
                density=4.0, #5.0,
                friction=0.1,
                categoryBits=0x0010,
                maskBits=0x0011,   # collide only with ground and lander
                restitution=0.0)  # 0.99 bouncy
                )            
        self.skycrane.color1 = (0.8, 0.4, 0.5)
        self.skycrane.color2 = (0.5, 0.3, 0.3)

        initial_force_applied = (
            self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM),
            self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM)
            )
        #print('initial angle', self.skycrane.angle)
        self.skycrane.ApplyForceToCenter( initial_force_applied , True)

        ########################
        # The Lander
        ########################
        initial_y = VIEWPORT_H/SCALE-TETHER_LENGTH/SCALE
        self.lander = self.world.CreateDynamicBody(
            position=(VIEWPORT_W/SCALE/2, initial_y),
            angle=0.0,
            fixtures = fixtureDef(
                shape=polygonShape(vertices=[(x/SCALE, y/SCALE) for x, y in LANDER_POLY]),
                density=5.0,
                friction=0.1,
                categoryBits=0x0010,
                maskBits=0x0011,   # collide only with ground and skycrane
                restitution=0.0)  # 0.99 bouncy
                )
        self.lander.color1 = (0.5, 0.4, 0.9)
        self.lander.color2 = (0.3, 0.3, 0.5)
        self.lander.ApplyForceToCenter( initial_force_applied, True)

        self.legs = []
        for i in [-1, +1]:
            leg = self.world.CreateDynamicBody(
                position=(VIEWPORT_W/SCALE/2 - i*LEG_AWAY/SCALE, initial_y),
                angle=(i * 0.05),
                fixtures=fixtureDef(
                    shape=polygonShape(box=(LEG_W/SCALE, LEG_H/SCALE)),
                    density=1.0,
                    restitution=0.0,
                    friction=0.8,
                    categoryBits=0x0010,
                    maskBits=0x0011)
                )
            leg.ground_contact = False
            leg.color1 = (0.5, 0.4, 0.9)
            leg.color2 = (0.3, 0.3, 0.5)
            rjd = revoluteJointDef(
                bodyA=self.lander,
                bodyB=leg,
                localAnchorA=(0, 0),
                localAnchorB=(i * LEG_AWAY/SCALE, LEG_DOWN/SCALE),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=LEG_SPRING_TORQUE,
                motorSpeed=+0.3 * i  # low enough not to jump back into the sky
                )
            if i == -1:
                rjd.lowerAngle = +0.9 - 0.5  # The most esoteric numbers here, angled legs have freedom to travel within
                rjd.upperAngle = +0.9
            else:
                rjd.lowerAngle = -0.9
                rjd.upperAngle = -0.9 + 0.5
            leg.joint = self.world.CreateJoint(rjd)
            self.legs.append(leg)

        ##########################
        # The Tether (Rope)
        #
        ##########################

        tetherdef = ropeJointDef(
            bodyA=self.lander,
            bodyB=self.skycrane,
            localAnchorA=(0, 17/SCALE),
            localAnchorB=(0,-10/SCALE),
            maxLength=TETHER_LENGTH/SCALE,
            #frequencyHz=FPS,
            #dampingRatio=0.2,
            collideConnected=True,
        )
        
        self.skycrane.joint = self.world.CreateJoint(tetherdef)
        self.tether_connected = 1

        # ##########################
        # # Dynamic Reward Indicators
        # ##########################

        # self.actual_reward_indicator = self.world.CreateDynamicBody(
        #     position=(VIEWPORT_W/SCALE/2, VIEWPORT_H-20/SCALE),
        #     angle=0.0,
        #     fixtures = fixtureDef(
        #         shape=polygonShape(vertices=[(x/SCALE, y/SCALE) for x, y in [
        #             (0,10),(10,0),(0,-10),(-10,0)
        #         ]]),
        #         density=1.0,
        #         friction=0.1,
        #         categoryBits=0x0000,
        #         maskBits=0x0000,   # collide only with ground and skycrane
        #         restitution=0.0)  # 0.99 bouncy
        #         )
        # self.actual_reward_indicator.color1 = (1.0, 0.1, 0.1)
        # self.actual_reward_indicator.color2 = (1.0, 0.1, 0.9)


        self.drawlist = [self.lander, self.skycrane] + self.legs

        return self.step(0)[0]

    def _create_particle(self, mass, x, y, ttl):
        p = self.world.CreateDynamicBody(
            position = (x, y),
            angle=0.0,
            fixtures = fixtureDef(
                shape=circleShape(radius=2/SCALE, pos=(0, 0)),
                density=mass,
                friction=0.1,
                categoryBits=0x0100,
                maskBits=0x001,  # collide only with ground
                restitution=0.3)
                )
        p.ttl = ttl
        self.particles.append(p)
        self._clean_particles(False)
        return p

    def _clean_particles(self, all):
        while self.particles and (all or self.particles[0].ttl < 0):
            self.world.DestroyBody(self.particles.pop(0))

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid " % (action, type(action))

        # Engines

        dispersion = [self.np_random.uniform(-1.0, +1.0) / SCALE for _ in range(2)]

        m_power = 0.0
        # 0 No operation
        # 1 Fire left engine full
        # 2 Fire both engines half
        # 3 Fire right engine full
        # 4 release tether

        sincos  = (math.sin(self.skycrane.angle), math.cos(self.skycrane.angle))
        #side = (-sincos[1], sincos[0])

        thruster_angles = [
            self.skycrane.angle+math.pi/2 - SIDE_ENGINE_ANGLE,
            self.skycrane.angle+math.pi/2 + SIDE_ENGINE_ANGLE
        ]

        thruster_vec = [
            [math.cos(thruster_angles[0])/SCALE, math.sin(thruster_angles[0])/SCALE],
            [math.cos(thruster_angles[1])/SCALE, math.sin(thruster_angles[1])/SCALE],
        ]
        if action in [1,2,3]:
            # was: Main engine
            # Fire both engines half


            # thruster positions
            # x = 
            # self.skycrane.position[0] # x in the middle of the ship
            #   +- SIDE_ENGINE_AWAY *cos(sc.angle) # depend on thruster side
            #   + SIDE_ENGINE_HEIGHT *sin(sc.angle)
            # y=
            # self.skycrane.position[1]
            #   +- SIDE_ENGINE_AWAY * -sin(sc.angle)
            #   + SIDE_ENGINE_HEIGHT *cos(sc.angle)

            # thruster directions
            # self.skycrane.angle+pi/2 -/+ SIDE_ENGINE_ANGLE


            if action == 2:
                m_power = 1.0 * HALF_ENGINE_POWER/FULL_ENGINE_POWER
            else:
                m_power = 1.0 # FULL POWER

            # ox,oy = Offset X, Y
            # engine 1
            ox1 = (sincos[0] * (SIDE_ENGINE_HEIGHT/SCALE + 1 * dispersion[0]) +
                  sincos[1] * (-SIDE_ENGINE_AWAY/SCALE + 1* dispersion[1]))  
            oy1 = -(sincos[1] * (SIDE_ENGINE_HEIGHT/SCALE + 1 * dispersion[0]) - 
                  sincos[0] * (-SIDE_ENGINE_AWAY/SCALE + dispersion[1]))
            # engline 2
            ox2 = (sincos[0] * (SIDE_ENGINE_HEIGHT/SCALE + 1 * dispersion[0]) +
                  sincos[1] * (SIDE_ENGINE_AWAY/SCALE + 1* dispersion[1]))  
            oy2 = -(sincos[1] * (SIDE_ENGINE_HEIGHT/SCALE + 1 * dispersion[0]) - 
                  sincos[0] * (SIDE_ENGINE_AWAY/SCALE + dispersion[1]))

            impulse_pos1 = (
                self.skycrane.position[0] + ox1, 
                self.skycrane.position[1] + oy1
                )
            impulse_pos2 = (
                self.skycrane.position[0] + ox2, 
                self.skycrane.position[1] + oy2
                )
            if action in [1,2]:
                p1 = self._create_particle(3.5,  # 3.5 is here to make particle speed adequate
                                        impulse_pos1[0],
                                        impulse_pos1[1],
                                        m_power)  # particles are just a decoration
            else:
                p1 = None
            if action in [2,3]:
                p2 = self._create_particle(3.5,  # 3.5 is here to make particle speed adequate
                                impulse_pos2[0],
                                impulse_pos2[1],
                                m_power)  # particles are just a decoration
            else:
                p2 = None

            for ox, oy, p,impulse_pos, thrust_vec in zip(
                [ox1,ox1],
                [oy1,oy2],
                [p1,p2],
                [impulse_pos1, impulse_pos2], 
                thruster_vec):
                if not p is None: # only fire engines activated with particles
                    p.ApplyLinearImpulse(
                        # impulse vector
                        ( 
                            -thrust_vec[0] * FULL_ENGINE_POWER * m_power, 
                            -thrust_vec[1] * FULL_ENGINE_POWER * m_power,
                            ),
                        impulse_pos,
                        True
                    )
                    self.skycrane.ApplyLinearImpulse(
                        (
                            thrust_vec[0] * FULL_ENGINE_POWER * m_power, 
                            thrust_vec[1] * FULL_ENGINE_POWER * m_power,),
                        impulse_pos,
                        True,
                    )

        s_power = 0.0
        tether_abuse = False
        if action==4: 
            # RELEASE TETHER
            if self.tether_connected==1:
                if self.legs[0].ground_contact+self.legs[1].ground_contact > 0:
                    self.world.DestroyJoint(self.skycrane.joint)
                    self.tether_connected=0
                    self.skycrane.color1 = (0.0, 0.9, 0.9)
                else:
                    tether_abuse = True

        self.world.Step(1.0/FPS, 6*30, 2*30)

        pos = self.skycrane.position
        vel = self.skycrane.linearVelocity
        pos_lander = self.lander.position
        vel_lander = self.lander.linearVelocity
        state = [
            # SkyCrane Data
            (pos.x - VIEWPORT_W/SCALE/2) / (VIEWPORT_W/SCALE/2), #0
            (pos.y - (self.helipad_y+(LEG_DOWN+TETHER_LENGTH*1.3)/SCALE)) / (VIEWPORT_H/SCALE/2), #1
            vel.x*(VIEWPORT_W/SCALE/2)/FPS, #2
            vel.y*(VIEWPORT_H/SCALE/2)/FPS, #3
            self.skycrane.angle, #4
            20.0*self.skycrane.angularVelocity/FPS, #5 Squared Angular velocity to punish rotation
            # Lander Data
            1.0 if self.legs[0].ground_contact else 0.0, #6
            1.0 if self.legs[1].ground_contact else 0.0, #7
            (pos_lander.x - VIEWPORT_W/SCALE/2) / (VIEWPORT_W/SCALE/2), #8
            (pos_lander.y - (self.helipad_y+LEG_DOWN/SCALE)) / (VIEWPORT_H/SCALE/2), #9
            vel_lander.x*(VIEWPORT_W/SCALE/2)/FPS, #10
            vel_lander.y*(VIEWPORT_H/SCALE/2)/FPS, #11
            self.lander.angle, #12
            20.0*self.lander.angularVelocity/FPS,     #13
            self.tether_connected,       #14
        ]
        assert len(state) == 15
        reward = 0
        shaping = ( 
            # penalty for lander  distance from helipad 
            - 100*np.sqrt(state[8]*state[8] + state[9]*state[9]) #type:ignore 
            
            # penalty for lander speed
            - 100*np.sqrt(state[10]*state[10] + state[11]*state[11]) #type:ignore  

            -  50*abs(state[4])  # penalty for skycrane angle
            -  70*abs(state[5]) # penalty for skycrane rotational velocity
            + 10*state[6]        # reward for leg ground contacts
            + 10*state[7]        # reward for leg ground contacts
            - 70*abs(state[12]) # penalty for lander angle
            - 100*abs(state[13]) # penalty for lander rotational speed
            - 50*np.sqrt(state[0]*state[0] + state[1]*state[1]) #type:ignore # penalty for skycrane distance from drop off position
            )

            # And ten points for legs contact, the idea is if you
            # lose contact again after landing, you get negative reward
        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping

        reward -= m_power*0.10  # less fuel spent is better,
        
        reward += -5*tether_abuse # penalty for releasing tether before lander is on ground
        #print(dir(self.actual_reward_indicator))
        #self.actual_reward_indicator.transform(((VIEWPORT_W+reward)/SCALE, self.actual_reward_indicator.position.y),0)
        #self.actual_reward_indicator.position.x = (VIEWPORT_W+reward)/SCALE
        self.last_reward = reward

        done = False

        ####################
        # Extra Game Over Conditions - Except for crashing into the Mars
        #
        ###################

        if abs(state[8]) >= 1.0: # sudden fly-away: if the lander is off-screen
            self.game_over = True

        if abs(self.lander.angle) >= math.pi/2: # lander would be destroy by rope
            self.game_over = True
        
        if abs(self.skycrane.angle) >= math.pi/2: # skycrane would be destroy by rope
            self.game_over = True


        ####################



        if self.game_over: 
            done = True
            reward = -100

        if not self.lander.awake:
            done = True
            reward = +200
            print('Landing Award granted')
            #print('Terminal velocity:', state[11])
        return np.array(state, dtype=np.float32), reward, done, {}

    def render(self, mode='human'):
        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)
            self.viewer.set_bounds(0, VIEWPORT_W/SCALE, 0, VIEWPORT_H/SCALE)
            

        for obj in self.particles:
            obj.ttl -= 0.15
            obj.color1 = (max(0.2, 0.2+obj.ttl), max(0.2, 0.5*obj.ttl), max(0.2, 0.5*obj.ttl))
            obj.color2 = (max(0.2, 0.2+obj.ttl), max(0.2, 0.5*obj.ttl), max(0.2, 0.5*obj.ttl))

        self._clean_particles(False)

        # RED PLANET BACKGROUND
        self.viewer.draw_polygon(
            [(0,0), (self.viewer.width,0), (self.viewer.width, self.viewer.height,0), (0,self.viewer.height)], 
            color=(0.8, 0, 0)) # thanks go to Nico

        for p in self.sky_polys:
            self.viewer.draw_polygon(p, color=(0, 0, 0))
       # Tether
        if self.tether_connected:
            self.viewer.draw_polyline([
                self.lander.GetWorldPoint((0,17/SCALE)), self.skycrane.GetWorldPoint((0,-10/SCALE))
            ], color=(0.8,0.1,0.0)
            )
        for obj in self.particles + self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is circleShape:
                    t = rendering.Transform(translation=trans*f.shape.pos)
                    self.viewer.draw_circle(f.shape.radius, 20, color=obj.color1).add_attr(t)
                    self.viewer.draw_circle(f.shape.radius, 20, color=obj.color2, filled=False, linewidth=2).add_attr(t)
                else:
                    path = [trans*v for v in f.shape.vertices]
                    self.viewer.draw_polygon(path, color=obj.color1)
                    path.append(path[0])
                    self.viewer.draw_polyline(path, color=obj.color2, linewidth=2)
 

        for x in [self.helipad_x1, self.helipad_x2]:
            flagy1 = self.helipad_y
            flagy2 = flagy1 + 50/SCALE
            self.viewer.draw_polyline([(x, flagy1), (x, flagy2)], color=(1, 1, 1))
            self.viewer.draw_polygon([(x, flagy2), (x, flagy2-10/SCALE), (x + 25/SCALE, flagy2 - 5/SCALE)],
                                     color=(0.8, 0.8, 0))

        # reward indicator
        if self.render_reward_indicator:
            x = self.last_reward + VIEWPORT_W/2/SCALE
            ind_y = 30/SCALE 
            self.viewer.draw_polygon([(x, ind_y-20/SCALE), (VIEWPORT_W/2/SCALE, ind_y-20/SCALE), (VIEWPORT_W/2/SCALE, ind_y)],
                                        color=(0.9, 0.9, 0.9))        

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


