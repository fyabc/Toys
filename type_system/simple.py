#! /usr/bin/python
# -*- encoding: utf-8 -*-

"""A simple type system."""

__author__ = 'fyabc'


class Type:
    """Class of types."""

    # The precedence, used in representation to add parentheses.
    precedence = 0

    # If self_paren, add parentheses when precedence equal to child's precedence.
    self_paren = False
    
    def __str__(self):
        raise NotImplementedError()
    
    def __repr__(self):
        return self.__str__()

    def _prec_str(self, child):
        if child.precedence < self.precedence or child.precedence == self.precedence and self.self_paren:
            return '({})'.format(child)
        return str(child)

    def is_primitive(self):
        return False

    def is_pointer(self):
        return False

    def is_list(self):
        return False

    def is_function(self):
        return False

    def is_tuple(self):
        return False

    def is_template(self):
        return False


class Primitive(Type):
    """Class of primitive types. They are singleton classes."""

    precedence = 100
    
    name = None
    
    def __init__(self):
        raise Exception('Cannot instantiate primitive types directly')
    
    @classmethod
    def get(cls):
        return cls.__new__(cls)
    
    def __str__(self):
        return self.name

    def is_primitive(self):
        return True
    
    
class Bool(Primitive):
    name = 'Bool'
    

class Int(Primitive):
    name = 'Int'
    
    
class Float(Primitive):
    name = 'Float'


class Pointer(Type):
    precedence = 70

    def __init__(self, pointee):
        self.pointee = pointee

    def __str__(self):
        return '{}*'.format(self._prec_str(self.pointee))

    def is_pointer(self):
        return True


class Function(Type):
    precedence = 50
    self_paren = True

    def __init__(self, args, ret):
        self.args = list(args)
        self.args.append(ret)

    def __str__(self):
        return ' -> '.join(self._prec_str(arg) for arg in self.args)

    def is_function(self):
        return True


class List(Type):
    precedence = 60

    def __init__(self, value_type: Type):
        self.value_type = value_type

    def __str__(self):
        return '[{}]'.format(self.value_type)

    def is_list(self):
        return True


class Tuple(Type):
    precedence = 60

    def __init__(self, value_types: list):
        self.value_types = value_types

    def __str__(self):
        return '({})'.format(', '.join(self._prec_str(value) for value in self.value_types))

    def is_tuple(self):
        return True


class Kind:
    def instantiate(self, *args, **kwargs):
        pass

    def __str__(self):
        return ''

    def __repr__(self):
        return self.__str__()


class Template(Kind):
    class TemplateType(Type):
        precedence = 60

        def __init__(self, name: str, args: list):
            self.name = name
            self.args = args

        def __str__(self):
            return '{}<{}>'.format(self.name, ', '.join(self._prec_str(arg) for arg in self.args))

        def is_template(self):
            return True

    def __init__(self, name: str, n_args: int):
        self.name = name
        self.n_args = n_args

    def instantiate(self, args: list):
        return self.TemplateType(self.name, args)

    def __str__(self):
        return '{}<{}>'.format(self.name, ', '.join('a{}'.format(i) for i in range(1, self.n_args + 1)))


def parse_type(type_str):
    """Parse the type string to the type."""

    pass


def test():
    from pprint import pprint

    types = {
        'f': Float.get(),
        'i': Int.get(),
        'b': Bool.get(),
    }

    types['pf'] = Pointer(types['f'])
    types['ppf'] = Pointer(types['pf'])
    types['func1'] = Function([Int.get(), Function([Float.get()], Int.get())], types['pf'])
    types['l_func1'] = List(types['func1'])
    types['func2'] = Function([types['l_func1'], Tuple([types['pf'], Int.get()])], Bool.get())
    types['p_func2'] = Pointer(types['func2'])

    vector = Template('Vector', 1)
    types['vi'] = vector.instantiate([Int.get()])
    types['vvi'] = vector.instantiate([types['vi']])

    pprint(types, indent=2)

    pprint(vector)


if __name__ == '__main__':
    test()
