# -*- coding: utf-8 -*-

from pprint import pprint
from pyparsing import *

__author__ = 'fyabc'

"""
Parsing rule:

form ::= form_name newline field+
field ::= field_name colon field_type [arrow property+]
property ::= key colon value
form_name ::= word
field_name ::= word
field_type ::= CharField | EmailField | PasswordField
key ::= word
value ::= alphanumeric+
word ::= alphanumeric+
newline ::= \n
colon ::= :
arrow ::= ->
"""


def makeUserForm():
    def properties2dict(tokens):
        result = {}
        for token in tokens:
            result[token.propertyKey] = token.propertyValue
        return result

    def fields2dict(tokens):
        result = {}
        for token in tokens:
            result[token.fieldName] = {
                'fieldType': token.fieldType,
                'properties': token.fieldProperties,
            }
        return result

    def forms2dict(tokens):
        result = {}
        for token in tokens:
            result[token.formName] = token[1]
        return result

    # Must let EOL pass.
    ParserElement.setDefaultWhitespaceChars(' \t')

    newLine = Suppress('\n')
    colon = Suppress(':')
    arrow = Suppress('->')
    word = Word(alphanums)

    key = word('propertyKey')
    value = word('propertyValue')

    fieldType = oneOf('CharField EmailField PasswordField')('fieldType')
    fieldName = word('fieldName')
    fieldProperty = Group(key + colon + value)('fieldProperty')
    field = Group(fieldName + colon + fieldType +
                  Optional(arrow + OneOrMore(fieldProperty)).setParseAction(properties2dict)('fieldProperties')
                  + newLine)('field')

    formName = word('formName')
    form = Group(formName + newLine + OneOrMore(field).setParseAction(fields2dict))
    forms = ZeroOrMore(form + ZeroOrMore(newLine))('forms').setParseAction(forms2dict)

    return forms


def main():
    forms = makeUserForm()

    inputStr = """\
    UserForm007
    name: CharField -> label: UserName size: 25
    email: EmailField -> size: 32
    password: PasswordField

    UserForm004
    id: CharField

    UserForm012
    email: EmailField -> address: Jintan
    firstName: CharField -> value: Yang
"""

    result = forms.parseString(inputStr, parseAll=True)

    def message(code, result=result):
        print(code + ':')
        print(eval(code))

    import json
    print(json.dumps(result.asDict(), indent=4))

    message('result')


if __name__ == '__main__':
    main()
