#! /usr/bin/python
# -*- encoding: utf-8 -*-

"""

"""

import numbers
import types
import operator
import collections

__author__ = 'fyabc'


# [NOTE]: The reversed operators in Python (e.g. `__rsub__`)
#   The reversed operators will NOT be taken into account when the two operands have SAME type.
#   The expression `a - b` may be interpreted as:
#
#       try:
#           return a.__sub__(b)
#       except TypeError as e_a:
#           if type(a) == type(b):
#               raise e_a
#           else:
#               try:
#                   return b.__rsub__(a)
#               except TypeError as e_b:
#                   raise e_b

# [NOTE]: The `__div__` is `__truediv__` and `__floordiv__` in Python 3.


class Expression:
    precedence = None
    operatorName = None
    eval_ = None

    def __init__(self, name=''):
        self.operands = []
        self.name = name

    def pprint(self, format_=None):
        raise NotImplementedError()

    def __add__(self, other):
        return add(self, other)

    def __radd__(self, other):
        return add(other, self)

    def __sub__(self, other):
        return sub(self, other)

    def __rsub__(self, other):
        return sub(other, self)

    def __mul__(self, other):
        return mul(self, other)

    def __rmul__(self, other):
        return mul(other, self)

    def __truediv__(self, other):
        return divide(self, other)

    def __rtruediv__(self, other):
        return divide(other, self)

    def __mod__(self, other):
        return modulus(self, other)

    def __rmod__(self, other):
        return modulus(other, self)

    def __pow__(self, power_, modulo=None):
        powered = power(self, power_)
        if modulo is None:
            return powered
        return modulus(powered, modulo)

    def __rpow__(self, other):
        return power(other, self)

    def eval(self, valueDict=None):
        pass

    def grad(self, variables):
        useSequence = isinstance(variables, collections.Sequence)
        if not useSequence:
            variables = [variables]

        result = []
        for variable in variables:
            if not isinstance(variable, Variable):
                raise TypeError('Expected Variable, got {} of type {}'
                                .format(str(variable), str(type(variable))))
            result.append(self._grad(variable))

        return result if useSequence else result[0]

    def _grad(self, variable):
        raise NotImplementedError()


class Variable(Expression):
    precedence = None

    def __init__(self, name=None):
        super(Variable, self).__init__(name)

    def pprint(self, format_=None):
        if self.name is None:
            return "Variable('')"
        return self.name

    def eval(self, valueDict=None):
        valueDict = valueDict or {}

        if self in valueDict:
            return valueDict[self]
        raise KeyError('The value of variable {} not given'.format(self.name))

    def _grad(self, variable):
        return Constant(1) if self == variable else Constant(0)


class Constant(Expression):
    precedence = None

    def __init__(self, value=0, name=None):
        super(Constant, self).__init__(name)
        self.value = value

    def pprint(self, format_=None):
        return str(self.value)

    def eval(self, valueDict=None):
        return self.value

    def _grad(self, variable):
        return Constant(0)


def toConstant(operand):
    if isinstance(operand, numbers.Number):
        return Constant(operand)
    return operand


class UnaryExpression(Expression):
    precedence = None
    operatorName = None
    eval_ = None

    def __init__(self, operand):
        super(UnaryExpression, self).__init__()
        self.operand = operand
        self.operands = [self.operand]

    def pprint(self, format_=None):
        raise NotImplementedError()

    def eval(self, valueDict=None):
        return self.eval_(self.operand.eval(valueDict))

    def _grad(self, variable):
        raise NotImplementedError()


class BinaryExpression(Expression):
    precedence = None
    operatorName = None
    commutative = False
    eval_ = None

    def __init__(self, lhs, rhs):
        super(BinaryExpression, self).__init__()
        self.lhs, self.rhs = lhs, rhs
        self.operands = [self.lhs, self.rhs]

    def pprint(self, format_=None):
        return '({} {} {})'.format(self.lhs.pprint(), self.operatorName, self.rhs.pprint())

    def eval(self, valueDict=None):
        return self.eval_(self.lhs.eval(valueDict), self.rhs.eval(valueDict))

    def _grad(self, variable):
        raise NotImplementedError()

    @classmethod
    def makeExpression(cls, className, precedence, operatorName, eval_, commutative=False, **otherFuncs):
        def __init__(self, lhs, rhs):
            return super(self.__class__, self).__init__(lhs, rhs)

        classDict = {
            '__init__': __init__,
            'precedence': precedence,
            'operatorName': operatorName,
            'commutative': commutative,
            'eval_': eval_,
        }

        classDict.update(otherFuncs)

        newOperation = types.new_class(className, (cls,), {}, lambda ns: ns.update(classDict))
        newOperation.__module__ = __name__

        return newOperation

    @classmethod
    def makeOperation(cls):
        def binaryExpression(lhs, rhs):
            lhs = toConstant(lhs)
            rhs = toConstant(rhs)
            return cls(lhs, rhs)

        return binaryExpression


Add = BinaryExpression.makeExpression(
    'Add', 10, '+', operator.add, commutative=True,
    _grad=lambda self, variable: self.lhs._grad(variable) + self.rhs._grad(variable)
)
add = Add.makeOperation()

Sub = BinaryExpression.makeExpression(
    'Sub', 10, '-', operator.sub,
    _grad=lambda self, variable: self.lhs._grad(variable) - self.rhs._grad(variable)
)
sub = Sub.makeOperation()

Multiply = BinaryExpression.makeExpression(
    'Multiply', 20, '*', operator.mul, commutative=True,
    _grad=lambda self, variable: self.lhs._grad(variable) * self.rhs + self.lhs * self.rhs._grad(variable)
)
mul = Multiply.makeOperation()

Divide = BinaryExpression.makeExpression(
    'Divide', 20, '/', operator.truediv,
    _grad=lambda self, variable:
        (self.lhs._grad(variable) * self.rhs / self.lhs * self.rhs._grad(variable)) / (self.rhs ** 2)
)
divide = Divide.makeOperation()

Modulus = BinaryExpression.makeExpression('Modulus', 20, '%', operator.mod)
modulus = Modulus.makeOperation()

Power = BinaryExpression.makeExpression(
    'Power', 40, '**', operator.pow,
)
power = Power.makeOperation()


def test():
    x = Variable('x')
    y = (3 + x) * (4 - x)

    print(y.pprint())
    print(y.eval({x: 1}))
    print(y.grad(x).pprint())


if __name__ == '__main__':
    test()
