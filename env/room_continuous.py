import gym
import numpy as np
from copy import deepcopy
from rooms import roomEnv

class ContRoomEnv(roomEnv):
    def __init__(self):
        super().__init__()
        self.action_space =  gym.spaces.box.Box( # x and y velocities
            low=np.array([-1, -1], dtype=np.float32),
            high=np.array([1,  1], dtype=np.float32),
        )
    
    def step(self, action):
        self.steps += 1
        self.event = ""

        new_position = self.position + 0.5*action
        traj = [self.position, new_position]
        collided, wall = self.detect_collision(traj, self.current_room)
        if not collided:
            self.position = deepcopy(new_position)
        else:
            self.event = f"hit {wall} wall of {self.current_room}"

        new_room = self.get_current_room()
        if self.current_room != new_room:
            self.current_room = new_room
            self.event = f"entered {new_room}"
        
        obs = self.get_obs()
        return obs, 0, False, {"event": self.event}  
    
if __name__ == "__main__":
    env = ContRoomEnv()
    for i in range(1000):
        a = np.random.uniform(-1,1, [2])
        _,_,_,info = env.step(a)
        print(info)
        env.render()