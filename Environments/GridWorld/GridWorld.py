import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from Environments.GridWorld.GridBoard import *
from os import path

action_set = {
    0: 'u',
    1: 'd',
    2: 'l',
    3: 'r',
}

class Gridworld:

    def __init__(self, size=4, mode='static'):
        if size >= 4:
            self.board = GridBoard(size=size)
        else:
            print("Minimum board size is 4. Initialized to size 4.")
            self.board = GridBoard(size=4)

        #Add pieces, positions will be updated later
        self.board.addPiece('Player','P',(0,0))
        self.board.addPiece('Goal','+',(1,0))
        self.board.addPiece('Pit','-',(2,0))
        self.board.addPiece('Wall','W',(3,0))

        self.mode = mode
        self.size = size

        if mode == 'static':
            self.initGridStatic()
        elif mode == 'player':
            self.initGridPlayer()
        else:
            self.initGridRand()

    def reset(self):
#        self.board = GridBoard(size=self.size)
#        self.board.addPiece('Player','P',(0,0))
#        self.board.addPiece('Goal','+',(1,0))
#        self.board.addPiece('Pit','-',(2,0))
#        self.board.addPiece('Wall','W',(3,0))
        if self.mode == 'static':
            self.initGridStatic()
        elif self.mode == 'player':
            self.initGridPlayer()
        else:
            self.initGridRand()
        return self.get_state()

    #Initialize stationary grid, all items are placed deterministically
    def initGridStatic(self):
        #Setup static pieces
        self.board.components['Player'].pos = (0,3) #Row, Column
        self.board.components['Goal'].pos = (0,0)
        self.board.components['Pit'].pos = (0,1)
        self.board.components['Wall'].pos = (1,1)

    #Check if board is initialized appropriately (no overlapping pieces)
    #also remove impossible-to-win boards
    def validateBoard(self):
        valid = True

        player = self.board.components['Player']
        goal = self.board.components['Goal']
        wall = self.board.components['Wall']
        pit = self.board.components['Pit']

        all_positions = [piece for name,piece in self.board.components.items()]
        all_positions = [player.pos, goal.pos, wall.pos, pit.pos]
        if len(all_positions) > len(set(all_positions)):
            return False

        corners = [(0,0),(0,self.board.size), (self.board.size,0), (self.board.size,self.board.size)]
        #if player is in corner, can it move? if goal is in corner, is it blocked?
        if player.pos in corners or goal.pos in corners:
            val_move_pl = [self.validateMove('Player', addpos) for addpos in [(0,1),(1,0),(-1,0),(0,-1)]]
            val_move_go = [self.validateMove('Goal', addpos) for addpos in [(0,1),(1,0),(-1,0),(0,-1)]]
            if 0 not in val_move_pl or 0 not in val_move_go:
                #print(self.display())
                #print("Invalid board. Re-initializing...")
                valid = False

        return valid

    #Initialize player in random location, but keep wall, goal and pit stationary
    def initGridPlayer(self):
        #height x width x depth (number of pieces)
        self.initGridStatic()
        #place player
        self.board.components['Player'].pos = randPair(0,self.board.size)

        if (not self.validateBoard()):
            #print('Invalid grid. Rebuilding..')
            self.initGridPlayer()

    #Initialize grid so that goal, pit, wall, player are all randomly placed
    def initGridRand(self):
        #height x width x depth (number of pieces)
        self.board.components['Player'].pos = randPair(0,self.board.size)
        self.board.components['Goal'].pos = randPair(0,self.board.size)
        self.board.components['Pit'].pos = randPair(0,self.board.size)
        self.board.components['Wall'].pos = randPair(0,self.board.size)

        if (not self.validateBoard()):
            #print('Invalid grid. Rebuilding..')
            self.initGridRand()

    def validateMove(self, piece, addpos=(0,0)):
        outcome = 0 #0 is valid, 1 invalid, 2 lost game
        pit = self.board.components['Pit'].pos
        wall = self.board.components['Wall'].pos
        new_pos = addTuple(self.board.components[piece].pos, addpos)
        if new_pos == wall:
            outcome = 1 #block move, player can't move to wall
        elif max(new_pos) > (self.board.size-1):    #if outside bounds of board
            outcome = 1
        elif min(new_pos) < 0: #if outside bounds
            outcome = 1
        elif new_pos == pit:
            outcome = 2

        return outcome

    def step(self, a):
        action = action_set[a]
        #need to determine what object (if any) is in the new grid spot the player is moving to
        #actions in {u,d,l,r}
        def checkMove(addpos):
            if self.validateMove('Player', addpos) in [0,2]:
                new_pos = addTuple(self.board.components['Player'].pos, addpos)
                self.board.movePiece('Player', new_pos)
        if action == 'u': #up
            checkMove((-1,0))
        elif action == 'd': #down
            checkMove((1,0))
        elif action == 'l': #left
            checkMove((0,-1))
        elif action == 'r': #right
            checkMove((0,1))
        else:
            pass
        state = self.get_state()
        reward = self.reward()
        pos_player = self.board.components['Player'].pos
        pos_goal = self.board.components['Goal'].pos
        pos_pit = self.board.components['Pit'].pos
        done = True if (pos_player == pos_goal or pos_player == pos_pit) else False
        return state, reward, done, None
    
    def get_state(self):
        return self.board.render_np()

    def reward(self):
        if (self.board.components['Player'].pos == self.board.components['Pit'].pos):
            return -10
        elif (self.board.components['Player'].pos == self.board.components['Goal'].pos):
            return 10
        else:
            return -1

    def render(self):
        # Get the board as an array
        board = self.board.render_np()
		# Plot the grid
        fig, axes = plt.subplots(figsize=(6, 6))
        step = 1./self.size
        # offsetX, offsetY = 0.125, 0.125
        offsetX, offsetY = 0.5/self.size, 0.5/self.size
        tangulos = []
        tangulos.append(patches.Rectangle((0,0),0.998,0.998,\
                                        facecolor='cornsilk',\
                                        edgecolor='black',\
                                        linewidth=2))
        for j in range(self.size):
            locacion = j * step
            # Crea linea horizontal en el rectangulo
            tangulos.append(patches.Rectangle(*[(0, locacion), 1, 0.008],\
                    facecolor='black'))
            # Crea linea vertical en el rectangulo
            tangulos.append(patches.Rectangle(*[(locacion, 0), 0.008, 1],\
                    facecolor='black'))
        for t in tangulos:
            axes.add_patch(t)
        # Plot player
        player = np.where(board[0] == 1)
        y, x = player[0][0], player[1][0]
        y = (self.size - 1) - y
        path_image_robot = path.join('Environments', 'GridWorld', 'images', 'robot.png')
        arr_img = plt.imread(path_image_robot, format='png')
        image_robot = OffsetImage(arr_img, zoom=0.5/(self.size**0.7))
        image_robot.image.axes = axes
        ab = AnnotationBbox(
            image_robot,
            [(x*step) + offsetX, (y*step) + offsetY],
            frameon=False)
        axes.add_artist(ab)
        # Plot the exit
        exit = np.where(board[1] == 1)
        y, x = exit[0][0], exit[1][0]
        y = (self.size - 1) - y
        path_image_exit = path.join('Environments', 'GridWorld', 'images', 'exit.png')
        arr_img = plt.imread(path_image_exit, format='png')
        image_salida = OffsetImage(arr_img, zoom=0.125/(self.size**0.7))
        image_salida.image.axes = axes
        ab = AnnotationBbox(
            image_salida,
            [(x*step) + offsetX, (y*step) + offsetY],
            frameon=False)
        axes.add_artist(ab)
        # Plot pit
        pit = np.where(board[2] == 1)
        y, x = pit[0][0], pit[1][0]
        y = (self.size - 1) - y
        path_image_pit = path.join('Environments', 'GridWorld', 'images', 'pit.png')
        arr_img = plt.imread(path_image_pit, format='png')
        image_robot = OffsetImage(arr_img, zoom=0.8/(self.size**0.7))
        image_robot.image.axes = axes
        ab = AnnotationBbox(
            image_robot,
            [(x*step) + offsetX, (y*step) + offsetY],
            frameon=False)
        axes.add_artist(ab)
        # Plot the wall
        wall = np.where(board[3] == 1)
        y, x = wall[0][0], wall[1][0]
        y = (self.size - 1) - y
        t = patches.Rectangle(*[(x*step,y*step), step,step], facecolor='black')
        axes.add_patch(t)
        # Erase axis
        axes.axis('off')
        plt.show()

    def close(self):
        pass