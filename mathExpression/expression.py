#! /usr/bin/python
# -*- encoding: utf-8 -*-

"""

"""

import math
import numbers
import types
import operator
import collections
from abc import ABCMeta, abstractmethod

from utils import _pprints

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


class Expression(metaclass=ABCMeta):
    precedence = None
    operatorName = None
    eval_ = None

    def __init__(self, name=''):
        self.operands = []
        self.name = name

    @abstractmethod
    def pprint(self, format_=None):
        raise NotImplementedError()

    def __pos__(self):
        return self

    def __neg__(self):
        return neg(self)

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
        return self.eval_(*[operand.eval(valueDict) for operand in self.operands])

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

    @classmethod
    def _basicSimplify(cls, simplified):
        if all(map(lambda operand: isinstance(operand, Constant), simplified)):
            return Constant(cls.eval_(*[operand.value for operand in simplified]))
        return cls(*simplified)

    def simplify(self):
        simplified = self._getSimplifiedChildren()

        return self._basicSimplify(simplified)

    def _getSimplifiedChildren(self):
        return [operand.simplify() for operand in self.operands]


class TerminalExpression(Expression, metaclass=ABCMeta):
    isTerminal = True

    def __init__(self, name=None):
        super(TerminalExpression, self).__init__(name)


class Variable(TerminalExpression):
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

    def simplify(self):
        return self


class Constant(TerminalExpression):
    precedence = None

    def __init__(self, value=0., name=None):
        super(Constant, self).__init__(name)
        self.value = value

    def pprint(self, format_=None):
        return str(self.value)

    def eval(self, valueDict=None):
        return self.value

    def _grad(self, variable):
        return Constant(0)

    def simplify(self):
        return self

    @staticmethod
    def hasValue(expression, value):
        return isinstance(expression, Constant) and expression.value == value


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

    def _grad(self, variable):
        raise NotImplementedError()

    @classmethod
    def makeExpression(cls, className, precedence, operatorName, eval_, **otherFuncs):
        def __init__(self, operand):
            super(self.__class__, self).__init__(operand)

        classDict = {
            '__init__': __init__,
            'precedence': precedence,
            'operatorName': operatorName,
            'eval_': eval_,
        }

        classDict.update(otherFuncs)

        newOperation = types.new_class(className, (cls,), {}, lambda ns: ns.update(classDict))
        newOperation.__module__ = __name__

        return newOperation

    @classmethod
    def makeOperation(cls):
        def unaryOperation(operand):
            operand = toConstant(operand)
            return cls(operand)

        return unaryOperation


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

    def _grad(self, variable):
        raise NotImplementedError()

    @classmethod
    def makeExpression(cls, className, precedence, operatorName, eval_, commutative=False, **otherFuncs):
        def __init__(self, lhs, rhs):
            super(self.__class__, self).__init__(lhs, rhs)

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
        def binaryOperation(lhs, rhs):
            lhs = toConstant(lhs)
            rhs = toConstant(rhs)
            return cls(lhs, rhs)

        return binaryOperation


# Unary operations.

Negate = UnaryExpression.makeExpression(
    'Negate', 100, '-', operator.neg,
    _grad=lambda self, variable: -self.operand._grad(variable),
    pprint=lambda self, format_=None: '(-{})'.format(self.operand.pprint(format_)),
)
neg = Negate.makeOperation()

Exponent = UnaryExpression.makeExpression(
    'Exponent', 100, 'exp', math.exp,
    _grad=lambda self, variable: self * self.operand._grad(variable),
    pprint=_pprints.functionCallPPrint,
)
exp = Exponent.makeOperation()

Ln = UnaryExpression.makeExpression(
    'Ln', 100, 'ln', math.log,
    _grad=lambda self, variable: self.operand._grad(variable) / self.operand,
    pprint=_pprints.functionCallPPrint,
)
ln = Ln.makeOperation()


# Binary operations.

def _AddSimplify(self):
    lhs_s, rhs_s = self._getSimplifiedChildren()

    if Constant.hasValue(lhs_s, 0.):
        return rhs_s
    if Constant.hasValue(rhs_s, 0.):
        return lhs_s
    return self._basicSimplify([lhs_s, rhs_s])

Add = BinaryExpression.makeExpression(
    'Add', 10, '+', operator.add, commutative=True,
    _grad=lambda self, variable: self.lhs._grad(variable) + self.rhs._grad(variable),
    simplify=_AddSimplify,
)
add = Add.makeOperation()

Sub = BinaryExpression.makeExpression(
    'Sub', 10, '-', operator.sub,
    _grad=lambda self, variable: self.lhs._grad(variable) - self.rhs._grad(variable),
)
sub = Sub.makeOperation()


def _MulSimplify(self):
    lhs_s, rhs_s = self._getSimplifiedChildren()

    if Constant.hasValue(lhs_s, 0.) or Constant.hasValue(rhs_s, 0.):
        return Constant(0)
    if Constant.hasValue(lhs_s, 1.):
        return rhs_s
    if Constant.hasValue(rhs_s, 1.):
        return lhs_s
    return self._basicSimplify([lhs_s, rhs_s])

Multiply = BinaryExpression.makeExpression(
    'Multiply', 20, '*', operator.mul, commutative=True,
    _grad=lambda self, variable: self.lhs._grad(variable) * self.rhs + self.lhs * self.rhs._grad(variable),
    simplify=_MulSimplify,
)
mul = Multiply.makeOperation()


def _DivideSimplify(self):
    lhs_s, rhs_s = self._getSimplifiedChildren()

    if Constant.hasValue(lhs_s, 0.):
        return Constant(0.)
    if Constant.hasValue(rhs_s, 1.):
        return lhs_s
    return self._basicSimplify([lhs_s, rhs_s])

Divide = BinaryExpression.makeExpression(
    'Divide', 20, '/', operator.truediv,
    _grad=lambda self, variable:
        (self.lhs._grad(variable) * self.rhs / self.lhs * self.rhs._grad(variable)) / (self.rhs ** 2),
    simplify=_DivideSimplify,
)
divide = Divide.makeOperation()

Modulus = BinaryExpression.makeExpression('Modulus', 20, '%', operator.mod)
modulus = Modulus.makeOperation()

Power = BinaryExpression.makeExpression(
    'Power', 40, '**', operator.pow,
    _grad=lambda self, variable:
        self * (ln(self.lhs) * self.rhs._grad(variable) + self.lhs._grad(variable) / self.lhs * self.rhs),
)
power = Power.makeOperation()


def test():
    x = Variable('x')
    y = x - x ** 2 * ln(1 + 1.0 / x)

    gy = y.grad(x)
    gys = gy.simplify()

    print(y.pprint())
    print(y.eval({x: 1}))
    print(gy.pprint())
    print(gys.pprint())


if __name__ == '__main__':
    test()
