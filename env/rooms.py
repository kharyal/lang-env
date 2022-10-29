import gym
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
# import keyboard

class roomEnv():
    def __init__(self):
        # self.action_space = gym.spaces.box.Box( # x-vel, y-vel
        #     low=np.array([-1, -1], dtype=np.float32),
        #     high=np.array([1, 1], dtype=np.float32),
        # )
        self.action_space = gym.spaces.Discrete(6)
        self.observation_space = gym.spaces.box.Box( # agent x,y, 4 objects' x,y and picked_objects
            low=np.array([-np.inf, -np.inf, -np.inf, -np.inf,-np.inf, -np.inf,-np.inf, -np.inf,-np.inf, -np.inf, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([np.inf,  np.inf, np.inf,  np.inf, np.inf,  np.inf, np.inf,  np.inf, np.inf,  np.inf, 1, 1, 1, 1], dtype=np.float32),
        )
        
        self.rooms = { #rooms are axis aligned rectangles defined by the top left and bottom right corners
            "main room": [(-10,10), (10,-10)],
            "northeast room": [(10,10), (18,2)],
            "northwest room": [(-18,10), (-10,2)],
            "southeast room": [(10,-2), (18,-10)],
            "southwest room": [(-18,-2), (-10,-10)],
        }
        self.walls = dict()
        for room in self.rooms.keys():
            self.walls[room] = []
            xs,ys = [self.rooms[room][0][0], self.rooms[room][1][0]], [self.rooms[room][0][1],self.rooms[room][1][1]]
            crnr = [[x,y] for x in xs for y in ys]
            self.walls[room].append([crnr[0],crnr[1]])
            self.walls[room].append([crnr[1],crnr[3]])
            self.walls[room].append([crnr[3],crnr[2]])
            self.walls[room].append([crnr[2],crnr[0]])
            self.walls[room] = np.array([self.walls[room]])
        self.gates = np.array([[[10,4],[10,8]], [[10,-4],[10,-8]], [[-10,4],[-10,8]], [[-10,-4],[-10,-8]],])

        self.wall_keys = ["east", "south", "west", "north"]
        self.action_keys = {
            0: np.array([0,1]),
            1: np.array([0,-1]),
            2: np.array([1,0]),
            3: np.array([-1,0]),
        }
        
        self.init_position = np.array([.0,.0])
        self.init_objects = np.array([
            ['black', [14,6]],
            ['green', [14,-6]],
            ['blue', [-14,-6]],
            ['yellow', [-14,6]],
        ])

        self.fig = None
        self.current_room = "main room"
        self.position = deepcopy(self.init_position)
        self.objects = deepcopy(self.init_objects)
        self.event = ""
        self.picked_objects = np.array([0,0,0,0], dtype=bool)

    def reset(self):
        self.current_room = "main room"
        self.position = deepcopy(self.init_position)
        self.objects = deepcopy(self.init_objects)
        self.event = ""
        self.picked_objects = np.array([0,0,0,0], dtype=bool)

        obs = self.get_obs()
        return obs, {"event": self.event}

    def step(self, action):
        self.event = ""
        if action<4: # movement
            a = self.action_keys[action]
            new_position = self.position + 0.500001*a
            traj = [self.position, new_position]
            collided, wall = self.detect_collision(traj, self.current_room)
            if not collided:
                self.position = deepcopy(new_position)
            else:
                self.event = f"hit {wall} wall of {self.current_room}"
            
            if np.any(self.picked_objects):
                i = np.argmax(self.picked_objects)
                self.objects[i][1] = deepcopy(self.position)

            new_room = self.get_current_room()
            if self.current_room != new_room:
                self.current_room = new_room
                self.event = f"entered {new_room}"
        
        elif action == 4 and not np.any(self.picked_objects): # pick
            min_dist = np.inf; min_dist_obj = None
            for i, object in enumerate(self.objects):
                dist = np.linalg.norm(np.array(object[1]) - self.position)
                if  dist <= 1/np.sqrt(2) and dist < min_dist:
                    min_dist = dist
                    min_dist_obj = i
            if min_dist_obj is not None:
                self.picked_objects[min_dist_obj] = True
                self.objects[min_dist_obj][1] = deepcopy(self.position)
                self.event = f"picked {self.objects[min_dist_obj][0]} object from {self.current_room}"
        
        elif action == 5 and np.any(self.picked_objects): # place
            i = np.argmax(self.picked_objects)
            self.picked_objects[i] = False
            self.event=f"placed {self.objects[i][0]} object in {self.current_room}"

        obs = self.get_obs()
        return obs, 0, False, {"event": self.event}        
            
    def get_current_room(self):
        min_area, min_area_room = np.inf, None
        for room in self.rooms.keys():
            x1,y1,x2,y2 = *self.rooms[room][0], *self.rooms[room][1]
            if x1<=self.position[0]<=x2 and y2<=self.position[1]<=y1:
                area = np.abs((x2-x1)*(y2-y1))
                if area < min_area:
                    min_area = area
                    min_area_room = room
        return min_area_room

    def get_obs(self):
        obs = np.hstack([self.position, *[i[1] for i in self.objects], self.picked_objects])
        return obs

    def ccw(self, A,B,C):
        return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
    def intersect(self,A,B,C,D):
        '''
        Check if lines AB and CD intersect
        '''
        return self.ccw(A,C,D) != self.ccw(B,C,D) and self.ccw(A,B,C) != self.ccw(A,B,D)
    
    def detect_collision(self, traj, current_room):
        for gate in self.gates:
            if self.intersect(traj[0], traj[1], gate[0], gate[1]):
                return False, None
        for i,wall in enumerate(self.walls[current_room][0]):
            if self.intersect(traj[0], traj[1], wall[0], wall[1]):
                return True, self.wall_keys[i]
        return False, None

    def render(self):
        if self.fig is None:
            self.fig,self.ax = plt.subplots()

        self.ax.cla()

        # render walls
        for room in self.rooms.keys():
            self.ax.plot(self.walls[room][0][:, 0, 0], self.walls[room][0][:, 0, 1], c = 'b')
            self.ax.plot(self.walls[room][0][-1,:,0], self.walls[room][0][-1,:,1], c='b')
        for gate in self.gates:
            self.ax.plot(gate[:,0], gate[:,1], c='white')
        self.ax.scatter(*self.position, c='r', s=60)
        for object in self.objects:
            self.ax.scatter(*object[1], c=object[0], s=20)
        plt.pause(0.0001)


if __name__ == "__main__":
    env = roomEnv()
    for i in range(200):
        action = 0
        if i < 12:
            action = 0
        elif i < 50:
            action = 2
        elif i < 75:
            action = 3
        elif i < 200:
            action = 3
        env.step(action)
        if i <75:
            env.step(4)
        else:
            env.step(5)
        if i == 100:
            env.reset()
        env.render()
