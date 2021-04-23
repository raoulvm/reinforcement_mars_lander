Based on openai.com's LunarLander [gym](https://github.com/openai/gym), the

# MARS LANDER

 is an increased challenge for reinforcement learning.

As [NASA did with Perseverance](https://en.wikipedia.org/wiki/Mars_2020), the lander this time has to be dropped of by a 
SkyCrane like EDLS - a rocket propelled crane platform that carries its payload to the surface,
to set it down safely and then leave the area.

Challenges:
* The state object is larger (lander and platform data)


    * SkyCrane Data
        *   x,              #0  -100%...100%
        *   y,              #1 -100%...100%
        *   velocity.x,     #2
        *   velocity.y, #   3
        *   skycrane.angle, #4
        *   angularVelocity, #5
    *  Lander Data
        *   legs[0].ground_contact , #6
        *   legs[1].ground_contact,  #7
        *   x ,  #8
        *   y ,  #9
        *   velocity.x,   #10
        *   velocity.y,   #11
        *   lander.angle,   #12
        *   lander.angularVelocity, #13
        *   tether_connected,       #14


* Phsyics are more demanding: The lander is an inert mass on a pendulum tether
* More actions
    * 0 No operation
    * 1 Fire left engine 
    * 2 Fire both engines 
    * 3 Fire right engine 
    * 4 release tether


## Requirements

You need to get [Box2d](https://box2d.org/) running on your envirnment. Easy on Linux, a challenge on Windows, and ah, duh, don't know about MacOS.
And you need open.ai Gym with `pip install gym` and tensorflow if you want to run the prepared reinforcement model.


