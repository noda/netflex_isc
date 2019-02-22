"""
netflex/src/netflex.py by Jens Brage (jens.brage@noda.se)

This file is part of the Python project Netflex, which implements a version of
the alternating direction method of multipliers algorithm [1] tailored towards
model predictive control [2]. Netflex was carried out as part of the research
project Netflexible Heat Pumps within the research and innovation programme
SamspEL, funded by the Swedish Energy Agency. Netflex is made available under
the ISC licence [3], see the file LICENSE.md.

[1](http://stanford.edu/~boyd/admm.html)
[2](https://en.wikipedia.org/wiki/Model_predictive_control)
[3](https://en.wikipedia.org/wiki/ISC_license)
"""

import cvxpy
import numpy
import pandas
import textwrap

class NetflexKeyError(KeyError):
    """
    Used instead of built-in exception, see KeyError.
    """

    pass

class NetflexRuntimeError(RuntimeError):
    """
    Used instead of built-in exception, see RuntimeError.
    
    Raised when cvxpy.Problem.status != 'optimal', or when Market.run fails to
    converge whithin max_iters number of iterations.
    """

    pass

class NetflexTypeError(TypeError):
    """
    Used instead of built-in exception, see TypeError.
    """

    pass

class NetflexValueError(ValueError):
    """
    Used instead of built-in exception, see ValueError.
    """

    pass

class ExpressionMixin(object):
    """
    Mixin for an interval of a time series of floating point numbers.
    Essentially a hack to avoid repeating things for Variable and Parameter.
    """

    def __init__(self, label, start, periods, **kwargs):
        """
        Create an interval based on the superclass.

        :param label: the name of the time series
        :param start: the location of the interval relative to the present
        :param periods: the intervalranges over range(start, start + periods)
        :param **kwargs: see the superclass
        :returns: an interval based on the superclass
        :raises NetflexTypeError: on nonsense
        :raises NetflexTypeError: on nonsense
        """

        if not isinstance(label, str):
            raise NetflexTypeError('Expected instance of str, recieved %r.' % label)
        if not isinstance(start, int):
            raise NetflexTypeError('Expected instance of int, recieved %r.' % start)
        super().__init__((periods, ), **kwargs)
        self.label = label
        self.start = start

    @property
    def periods(self):
        """
        Provided for convenience.
        """

        periods, = self.shape
        return periods

    def __repr__(self):
        """
        Used to provide detailed error messages.
        """

        s = self.__class__.__name__ +'('
        s += '%r,' % self.label
        s += ' %r,' % self.start
        s += ' %r,' % self.periods
        s += ' **_)'
        return s

class Variable(ExpressionMixin, cvxpy.Variable):
    """
    Interval of a time series of floating point numbers for output. Essentially
    a cvxpy.Variable of ndim == 1 with attributes that locates it within a time
    series.
    """

    def __init__(self, label, start, periods, **kwargs):
        """
        Create a Variable, see ExpressionMixin.
        """
        
        ExpressionMixin.__init__(self, label, start, periods, **kwargs)

class Parameter(ExpressionMixin, cvxpy.Parameter):
    """
    Interval of a time series of floating point numbers for input. Essentially
    a cvxpy.Parameter of ndim == 1 with attributes that locates it within a
    time series.
    """

    def __init__(self, label, start, periods, **kwargs):
        """
        Create a Parameter, see ExpressionMixin.
        """

        ExpressionMixin.__init__(self, label, start, periods, **kwargs)

class Problem(cvxpy.Problem):
    """
    Local optimization problem with support for loading data from a data frame
    to the constituent parameters and for saving data from the constituent
    variables to a data frame. Note that while the problem can contain other
    cvxpy.Variable and cvxpy.Parameter, only the Variable and the Parameter are
    subject to the saving and loading of data.
    """

    def get_values(self, dataframe, start):
        """
        Save data from the variables to the data frame.

        :param dataframe: the data frame to save data to
        :param start: the location to be used as the present
        :raises NetflexTypeError: on nonsense
        """

        if not isinstance(dataframe, pandas.DataFrame):
            raise NetflexTypeError('Expected instance of pandas.DataFrame, recieved %r.' % dataframe)
        for o in self.variables():
            if isinstance(o, Variable):
                l = o.label
                if l in dataframe:
                    i = start + o.start
                    dataframe[l].values[i : i + o.periods] = o.value

    def set_values(self, dataframe, start):
        """
        Load data from the data frame to the parameters. Note that the
        Parameter.value can be initialized in different ways, and that it is
        possible to leave out the ADMM shadow prices from the data frame and
        instead rely on the default Parameter.value for a cold start.

        :param dataframe: the data frame to load data from
        :param start: the location to be used as the present
        :raises NetflexTypeError: on nonsense
        """

        if not isinstance(dataframe, pandas.DataFrame):
            raise NetflexTypeError('Expected instance of pandas.DataFrame, recieved %r.' % dataframe)
        for o in self.parameters():
            if isinstance(o, Parameter):
                l = o.label
                if l in dataframe:
                    i = start + o.start
                    o.value = dataframe[l].values[i : i + o.periods]
                else:
                    if o.value is None:
                        o.value = numpy.zeros(o.shape)

class Agent(object):
    """
    Encodes a local optimization problem, complete with termination criteria
    and penalty terms. Also, logs intermediate residuals.
    """

    # A shdow price has label == PREFIX + label of the corresponding quantity.
    PREFIX = '_'
    
    # Netflex implements a symmetric version of the ADMM algorith. Change the
    # parameters to try other versions, e.g., STEP = 0.0, 1.0 for the version
    # preferred by Boyd. 
    STEP = 1.0 / numpy.sqrt(2.0), 1.0 / numpy.sqrt(2.0)
    
    # In Boyd's notation, x == z, or SIGN[0] * x + SIGN[1] * z == 0.
    SIGN = 1.0, -1.0

    def __init__(self, part, *args, cost=None, constraints=None, **kwargs):
        """
        Create an Agent, complete with a log Agent.log over the residuals of
        intermediate computations.

        :param part: an int in 0, 1 signifying part in the implied bipartite graph
        :param args: a sequence of tuples (dx, dy, x0) where
            dx: a float > 0.0 used for penalty and the termination criteria
            dy: a float > 0.0 used for penalty
            x0: a Variable
        :param cost: the objective function to minimize
        :param constraints: a list of cvxpy.boolean
        :param kwargs: keyword arguments for cvxpy.Problem.solve
        :returns: an Agent
        :raises NetflexTypeError: on nonsense
        :raises NetflexValueError: on nonsense
        """

        if not isinstance(part, int):
            raise NetflexTypeError('Expected instance of int, recieved %r.' % part)
        if not (part in (0, 1)):
            raise NetflexValueError('Expected value in (0, 1), recieved %r.' % part)
        self.part = part
        self.args = []
        if cost is None:
            cost = 0.0
        if constraints is None:
            constraints = []
        for dx, dy, x0 in args:
            if not isinstance(dx, float):
                raise NetflexTypeError('Expected instance of float, recieved %r.' % dx)
            if not (numpy.isfinite(dx) and dx > 0.0):
                raise NetflexValueError('Expected finite value > 0, recieved %r.' % dx)
            if not isinstance(dy, float):
                raise NetflexTypeError('Expected instance of float, recieved %r.' % dy)
            if not (numpy.isfinite(dy) and dy > 0.0):
                raise NetflexValueError('Expected finite value > 0, recieved %r.' % dy)
            if not isinstance(x0, Variable):
                raise NetflexTypeError('Expected instance of Variable, recieved %r.' % x0)
            sp = x0.start, x0.periods
            y0 = Variable (Agent.PREFIX + x0.label, *sp)
            x1 = Parameter(               x0.label, *sp)
            y1 = Parameter(Agent.PREFIX + x0.label, *sp)
            self.args += [(dx, dy, x0, y0, x1, y1)]
            cost += Agent.SIGN[self.part] * x0 * y1 + dy / dx / 2.0 * cvxpy.sum_squares(x0 - x1)
            constraints += [y0 == y1 + dy / dx * Agent.STEP[self.part] * Agent.SIGN[self.part] * (x0 - x1)]
        self.problem = Problem(cvxpy.Minimize(cost), constraints)
        self.kwargs = kwargs 
        self.log = {}

    def __repr__(self):
        """
        Used to provide detailed error messages.
        """

        s = 'Agent(\n'
        s += '\t%r,\n' % self.part
        for dx, dy, x0, y0, x1, y1 in self.args:
            s += '\t(%r,' % dx + ' %r,' % dy + ' %r),\n' % x0
        s += '\tcost=_,\n'
        s += '\tconstraints=_,\n'
        for kw, arg in self.kwargs.items():
            s+= '\t%s=' % kw + '%r,\n' % arg
        s += ')'
        return s

    def get_values(self, dataframe, start):
        """
        Save data from the variables to the data frame, see Problem.get_values.
        """
        
        self.problem.get_values(dataframe, start)

    def set_values(self, dataframe, start):
        """
        Load data from the data frame to the parameters, see
        Problem.set_values.
        """
        
        self.problem.set_values(dataframe, start)

    def solve(self):
        """
        Perform one iteration in the search for a local solution in agreement
        with the other local solutions, and return a residual that measures how
        far the solution is from passing the constituent termination criteria.
        The latter is based on the max-norm rather than the l2-norm, and in
        addition simplified to only consider the primary residual, and not the
        secondary residual. While the use of the max-norm rather than the
        l2-norm is mathematically sound, the absence of the secondary residual
        can not be defended on mathematical grounds, though it works well
        enough in practise. Also, log the intermediate residuals.

        :returns: a float >= 0.0 with value <= 1.0 signifying an ok solution
        :raises NetflexRuntimeError: on failure to find a local solution
        """

        self.problem.solve(**self.kwargs)
        if self.problem.status != 'optimal':
            raise NetflexRuntimeError('Failed to converge, %s.' % self.problem.status)
        residual = 0.0
        for dx, dy, x0, y0, x1, y1 in self.args:
            r = max(abs(x0.value - x1.value)) / dx
            l = x0.label
            if self.log is not None:
                if l not in self.log:
                    self.log[l] = []
                self.log[l].append(r)
            residual = max(residual, r)
        return residual

class Exchange(Agent):
    """
    Encodes a local optimization problem tailored towards ADMM Exchange,
    complete with termination criteria and penalty terms. The optimization
    problem can also be solved analytically.
    """

    def __init__(self, part, *args, **kwargs):
        """
        Create an Exchange, see Agent.__init__.
        """

        c = [sum((x0 for dx, dy, x0 in args)) == cvxpy.Constant(0.0)]
        super().__init__(part, *args, constraints=c, **kwargs)

class Market(object):
    """
    Encodes a global optimization problem, complete with termination criteria
    and I_/O. Also, logs intermediate residuals.
    """

    def __init__(self, *agents):
        """
        Create a Market. The single-asterisk of the argument was choosen to
        permit that a sequence of agents be used as a component, for example,
        Market(*component_0, *component_1, ...).

        :param agents: a sequence of agents
        :returns: a Market
        :raises NetflexTypeError: on nonsense
        :raises NetflexValueError: when the agents fails to pass conditions
        necessary but not sufficient to form a bipartite graph; for conditions
        sufficient to form a bipartide graph, the agents must also pass
        Market.solve
        """

        self.partition = ([], {}), ([], {})
        for a0 in agents:
            try:
                if not isinstance(a0, Agent):
                    raise NetflexTypeError('Expected instance of Agent, recieved %r.' % a0)
                p0, p1 = self.partition[a0.part]
                p0 += [a0]
                for dx, dy, x0, y0, x1, y1 in a0.args:
                    l = x0.label
                    if l in p1:
                        raise NetflexValueError('Expected key not in dict, but %r in dict.' % l)
                    p1[l] = x1, y1
            except Exception as e:
                raise type(e)(str(e)[1 : -1] + ' Blame %r.' % a0)
        self.log = {}

    def __repr__(self):
        """
        Used to provide detailed error messages.
        """

        s = 'Market(\n'
        for i in range(2):
            for a0 in self.partition[i][0]:
                s += textwrap.indent('%r,\n' % a0, '\t')
        s += ')'
        return s

    def get_values(self, dataframe, start):
        """
        Save data from the variables to the data frame, see Problem.get_values.
        The method re-raises exceptions to the end of blaming the responsible
        agent, which helps when debugging multiagent systems.
        """

        for i in range(2):
            try:
                for a0 in self.partition[i][0]:
                    a0.get_values(dataframe, start)
            except Exception as e:
                raise type(e)(str(e) + ' Blame %r.' % a0)

    def set_values(self, dataframe, start):
        """
        Load data from the data frame to the parameters, see
        Problem.set_values. The method re-raises exceptions to the end of
        blaming the responsible agent, which helps when debugging multiagent
        systems.
        """

        for i in range(2):
            try:
                for a0 in self.partition[i][0]:
                    a0.set_values(dataframe, start)
            except Exception as e:
                raise type(e)(str(e) + ' Blame %r.' % a0)

    def solve(self):
        """
        Perform one iteration in the search for a global solution, and return a
        residual that measures how far the solution is from passing the
        constituent termination criteria. Also, log the intermediate
        residuals.

        :returns: a float >= 0.0 with value <= 1.0 signifying an ok solution
        :raise NetflexKeyError: when the agents fail to form a bipartite graph,
        and, as a consequence, the message passing breaks down
        """

        residual = 0.0
        for i in range(2):
            for a0 in self.partition[i][0]:
                try:
                    r = a0.solve()
                    l = a0.__class__.__name__ + '_%s' % id(a0)
                    if self.log is not None:
                        if l not in self.log:
                            self.log[l] = []
                        self.log[l].append(r)
                    residual = max(residual, r)
                    for dx, dy, x0, y0, x_, y_ in a0.args:
                        l = x0.label
                        try:
                            x1, y1 = self.partition[1 - i][1][l]
                        except KeyError as e:
                            raise NetflexKeyError('Expected key in dict, but %r not in dict.' % l)
                        x1.value = x0.value
                        y1.value = y0.value
                except Exception as e:
                    raise type(e)(str(e) + ' Blame %r.' % a0)
        return residual

    def run(self, dataframe, start, max_iters=-1):
        """
        Perform one iteration of model predictive control, that is, load the
        data from the data frame, search for a global solution, manage failure
        to find a global solution, and save the data to the data frame.

        :param dataframe: the data frame to load data from and save data to
        :param start: the location to be used as the present
        :param max_iters: the maximum number of iterations, or -1 for unbounded
        :raises NetflexTypeError: on nonsense
        :raises NetflexValueError: on nonsense
        :raises NetflexRuntimeError: on failure to find a global solution
        """

        if not isinstance(max_iters, int):
            raise NetflexTypeError('Expected instance of int, recieved %r.' % max_iters)
        if not (max_iters >= -1):
            raise NetflexValueError('Expected value >= -1, recieved %r.' % max_iters)
        self.set_values(dataframe, start)
        residual = 0.0
        num_iters = 0
        while num_iters != max_iters:
            residual = self.solve()
            if residual <= 1.0:
                break
            num_iters += 1
        if residual > 1.0:
            raise NetflexRuntimeError('Failed to converge for start == %r' % start + ' and max_iters == %r.' % max_iters)
        self.get_values(dataframe, start)

    def dot(self, path):
        """
        Produce a representation of the graph in the DOT language, see
        https://en.wikipedia.org/wiki/DOT_(graph_description_language).

        :param path: the path for file for output
        """

        table = str.maketrans({'/': 'p'})
        s = 'digraph %s {\n' % self.__class__.__name__
        for i in range(2):
            for j, a0 in enumerate(self.partition[i][0]):
                node_name = a0.__class__.__name__ + '_%s' % id(a0)
                s += '\t' + node_name + ' [shape=%s];\n' % ('house', 'invhouse')[i]
                for o in a0.problem.variables():
                    if type(o) is Variable:
                        edge_name = o.label.translate(table)
                        s += '\t' + edge_name + ' [shape=circle];\n'
                        s += '\t' + node_name + ' -> ' + edge_name + ';\n'
                for o in a0.problem.parameters():
                    if type(o) is Parameter:
                        edge_name = o.label.translate(table)
                        s += '\t' + edge_name + ' [shape=circle];\n'
                        s += '\t' + edge_name + ' -> ' + node_name + ';\n'
        s += '}\n'
        file = open(path, 'w')
        file.write(s)
        file.close()

def rolling_integral(x0, periods, function=None):
    """
    Integrate a function over a rolling window ending in the interval.

    :param x0: Variable or Parameter; the interval under consideration
    :param periods: the width of the rolling window
    :param function: a function from float to float to be integrated 
    :returns: cvxpy.Expression representing a vector of consecutive integrals
    """

    x = cvxpy.hstack([Parameter(x0.label, x0.start - periods, periods), x0])
    if function is None:
        f = x
        F = cvxpy.cumsum(f)
        return F[periods : x0.periods + periods] - F[0 : x0.periods]
    else:
        f = function(x)
        F = []
        for i in range(x0.periods):
            F += [cvxpy.cumsum(f[i : i + periods])]
        return cvxpy.hstack(F)
