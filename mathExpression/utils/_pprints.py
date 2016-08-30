#! /usr/bin/python
# -*- encoding: utf-8 -*-

__author__ = 'fyabc'


def functionCallPPrint(self, format_=None):
    operandsPPrint = ', '.join(map(lambda operand: operand.pprint(format_), self.operands))

    if len(self.operands) == 1 and hasattr(self.operands[0], 'isTerminal'):
        formatString = '{}({})'
    else:
        formatString = '{}{}'
    return formatString.format(self.operatorName, operandsPPrint)
