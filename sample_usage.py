import env.rooms as rooms

env = rooms.roomEnv()

for i in range(1000):
    if i%100 == 0:
        obs,info = env.reset()
    
    a = env.action_space.sample()
    obs, r, done, info = env.step(a)
    env.render()