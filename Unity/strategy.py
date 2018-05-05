import random

# class encompasses an exploration strategy -- general, minimalist template
class ExplorationStrategy:
    def __init__(self, action_size, args):
        self.action_size = action_size
        self.args = args

    #generalized update function to pass state to a generator, intended for real time updates
    def update(self, args):
        pass

    #non terminating generator that returns an action on each call
    def generate_trajectory(self, args):
        pass


######## example drone exploration strategies below ########


# alternates variable length arcs
class RandomArcStrategy(ExplorationStrategy):
    def __init__(self, action_size, args):
        ExplorationStrategy.__init__(self, action_size, args)
        # set up arc
        self.mean_turn_period = 3 # mean straight moves in between each turn action + 1
        self.sigma_turn_period = 5 # stddev straight moves in between each turn action + 1
        self.mean_arc_length = 20 # mean steps per arc
        self.sigma_arc_length = 10 # stddev steps per arc
        self.left_prob = 0.4 # probability the arc goes left

        self.update(self.args)

    #can be used like this as well
    def update(self, args):

        #specifies # straight moves for every turn (i.e, like radius)
        if "mean_turn_period" in args:
            self.mean_turn_period = args["mean_turn_period"]
        
        if "sigma_turn_period" in args:
            self.sigma_turn_period = args["sigma_turn_period"]

        if "mean_arc_length" in args:
            self.mean_arc_length = args["mean_arc_length"]

        if "base_arc_length" in args:
            self.base_arc_length = args["base_arc_length"]

        if "left_prob" in args:
            self.left_prob = args["left_prob"]

    #returns a generator
    def generate_trajectory(self, args):
        curr_length = 0

        while True:

            #new arc
            turn_period = max(1, random.gauss(mu=self.mean_turn_period, sigma=self.sigma_turn_period))
            arc_length = max(1, random.gauss(mu=self.mean_arc_length, sigma=self.sigma_arc_length))

            #left 
            directed_action = 0 if random.random() < self.left_prob else 2

            while curr_length < arc_length:
                if curr_length % turn_period == 0:
                    yield directed_action
                else:
                    #straight
                    yield 1



# alternates variable length arcs, but epsilon greedy (based on model)
class EGreedyArcStrategy(ExplorationStrategy):
    def __init__(self, action_size, args):
        ExplorationStrategy.__init__(self, action_size, args)
        # set up arc
        self.mean_turn_period = 3 # mean straight moves in between each turn action + 1
        self.sigma_turn_period = 5 # stddev straight moves in between each turn action + 1
        self.mean_arc_length = 20 # mean steps per arc
        self.sigma_arc_length = 10 # stddev steps per arc
        self.base_left_prob = 0.4 # probability the arc goes left
        self.best_action = 0 # the current best action (0 or 2)
        self.epsilon = 0.9 # for picking between best and random

        self.update(self.args)

    # must be called at each timestep to correctly update best_action
    def update(self, args):

        #specifies # straight moves for every turn (i.e, like radius)
        if "mean_turn_period" in args:
            self.mean_turn_period = args["mean_turn_period"]
        
        if "sigma_turn_period" in args:
            self.sigma_turn_period = args["sigma_turn_period"]

        if "mean_arc_length" in args:
            self.mean_arc_length = args["mean_arc_length"]

        if "base_arc_length" in args:
            self.base_arc_length = args["base_arc_length"]

        if "base_left_prob" in args:
            self.base_left_prob = args["base_left_prob"]

        if "best_action" in args:
            self.best_action = args["best_action"]

        if "epsilon" in args:
            self.epsilon = args["epsilon"]


    #returns a generator
    def generate_trajectory(self, args):
        curr_length = 0

        while True:

            #new arc
            turn_period = max(1, random.gauss(mu=self.mean_turn_period, sigma=self.sigma_turn_period))
            arc_length = max(1, random.gauss(mu=self.mean_arc_length, sigma=self.sigma_arc_length))

            #left 
            base_directed_action = 0 if random.random() < self.base_left_prob else 2

            if random.random() < self.epsilon:
                directed_action = best_action
            else:
                directed_action = base_directed_action

            while curr_length < arc_length:
                if curr_length % turn_period == 0:
                    yield directed_action
                else:
                    #straight
                    yield 1
