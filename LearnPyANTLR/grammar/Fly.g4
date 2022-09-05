grammar Fly;

// Parser

program
    : expression_list
    ;

expression_list
    : terminator? expression terminator?
    | expression_list expression terminator
    | terminator
    ;

expression
    : function_def
    | break_expression
    | continue_expression
    | rvalue
    ;

function_def
    : decorator* function_def_header function_def_body
    ;

decorator
    : AT dotted_name (LEFT_PAREN argument_list RIGHT_PAREN)? terminator
    ;

dotted_name
    : ID (DOT ID)*
    ;

function_def_header
    : DEF ID LEFT_PAREN parameter_list? RIGHT_PAREN
    ;

parameter_list
    : parameter (COMMA parameter)* COMMA?
    ;

parameter
    : ID
    | ID ASSIGN rvalue
    ;

function_def_body
    : LEFT_BRACE expression_list? RIGHT_BRACE
    | ASSIGN expression
    ;

rvalue
    : atom_expr
    | function_call
    | lambda_def
    | rvalue EXP rvalue
    | (NOT | BIT_NOT) rvalue
    | rvalue ( MUL | DIV | MOD ) rvalue
    | rvalue ( PLUS | MINUS ) rvalue
    | rvalue ( BIT_SHL | BIT_SHR ) rvalue
    | rvalue BIT_AND rvalue
    | rvalue ( BIT_OR | BIT_XOR ) rvalue
    | rvalue ( LESS | GREATER | LESS_EQUAL | GREATER_EQUAL ) rvalue
    | rvalue ( EQUAL | NOT_EQUAL ) rvalue
    | rvalue ( OR | AND ) rvalue
    | assign
    | statement
    ;

atom_expr
    : atom trailer*
    ;

atom
    : ID
    | STRING | NUMBER | BOOLEAN | NIL
    | list_literal | dict_set_literal | generator_literal
    ;

trailer
    : DOT ID
    | LEFT_BRACK rvalue RIGHT_BRACK
    ;


lvalue
    : ID
    | lvalue DOT ID
    | lvalue LEFT_BRACK rvalue RIGHT_BRACK
    ;

lambda_def
    : DO LEFT_PAREN parameter_list? RIGHT_PAREN function_def_body
    | DO function_def_body
    ;

statement
    : if_statement
    | for_statement
    | while_statement
    ;

if_statement
    : IF cond_expression terminator? statement_body
    | IF cond_expression terminator? statement_body ELSE statement_body
    | IF cond_expression terminator? statement_body elif_statement
    ;

elif_statement
    : ELIF cond_expression terminator? statement_body
    | ELIF cond_expression terminator? statement_body ELSE statement_body
    | ELIF cond_expression terminator? statement_body elif_statement
    ;

cond_expression
    : comp_expression
    | NOT cond_expression
    | cond_expression AND cond_expression
    | cond_expression OR cond_expression
    ;

comp_expression
    : rvalue (comp_op rvalue)*
    ;

comp_op
    : LESS | LESS_EQUAL | GREATER | GREATER_EQUAL | EQUAL | NOT_EQUAL | IN | NOT IN | IS | IS NOT
    ;

for_statement
    : FOR loop_vars IN rvalue terminator? statement_body
    ;

while_statement
    : WHILE cond_expression terminator? statement_body
    ;

loop_vars
    : ID (COMMA ID)*
    ;

statement_body
    : LEFT_BRACE expression_list? RIGHT_BRACE
    | expression
    ;

break_expression
    : BREAK
    ;

continue_expression
    : CONTINUE
    ;

function_call
    : function_call_wo_block lambda_def?
    ;

function_call_wo_block
    : callee LEFT_PAREN argument_list RIGHT_PAREN
    | callee LEFT_PAREN RIGHT_PAREN
    | callee argument_list
    ;

callee
    : atom_expr
    ;

argument_list
    : argument (COMMA argument)* COMMA?
    ;

argument
    : rvalue
    | ID ASSIGN rvalue
    ;

assign
    : lvalue ASSIGN rvalue
    | lvalue (PLUS_ASSIGN | MINUS_ASSIGN | MUL_ASSIGN | DIV_ASSIGN | MOD_ASSIGN | EXP_ASSIGN) rvalue
    ;

terminator
    : terminator SEMICOLON
    | terminator CRLF
    | SEMICOLON
    | CRLF
    | EOF
    ;

// TODO: Implement dict / list / set literals.
list_literal
    : LEFT_BRACK terminator? list_comp? terminator? RIGHT_BRACK
    ;

dict_set_literal
    : LEFT_BRACE terminator? RIGHT_BRACE
    ;

generator_literal
    : LEFT_PAREN terminator? list_comp? terminator? RIGHT_PAREN
    ;

list_comp
    : rvalue FOR loop_vars IN rvalue
    ;

dict_comp
    :
    ;

// Fragment & Lexer

/// Numeric literals.

NUMBER
    : INTEGER
    | FLOAT_NUMBER
    | IMAG_NUMBER
    ;

INTEGER
    : DECIMAL_INTEGER
    | OCT_INTEGER
    | HEX_INTEGER
    | BIN_INTEGER
    ;

DECIMAL_INTEGER
    : NON_ZERO_DIGIT DIGIT*
    | '0'+
    ;

OCT_INTEGER
    : '0' [oO] OCT_DIGIT+
    ;

HEX_INTEGER
    : '0' [xX] HEX_DIGIT+
    ;

BIN_INTEGER
    : '0' [bB] BIN_DIGIT+
    ;

FLOAT_NUMBER
    : POINT_FLOAT
    | EXPONENT_FLOAT
    ;

IMAG_NUMBER
    : ( FLOAT_NUMBER | INT_PART ) [jJ]
    ;

fragment NON_ZERO_DIGIT
    : [1-9]
    ;

fragment DIGIT
    : [0-9]
    ;

fragment OCT_DIGIT
    : [0-7]
    ;

fragment HEX_DIGIT
    : [0-9a-fA-F]
    ;

fragment BIN_DIGIT
    : [01]
    ;

fragment POINT_FLOAT
    : INT_PART? FRACTION
//    | INT_PART '.'
    ;

fragment EXPONENT_FLOAT
    : ( INT_PART | POINT_FLOAT ) EXPONENT
    ;

fragment INT_PART
    : DIGIT+
    ;

fragment FRACTION
    : '.' DIGIT+
    ;

fragment EXPONENT
    : [eE] [+-]? DIGIT+
    ;

/// String literals.

STRING
    : STRING_LITERAL
    ;

// TODO: Bytes literal, from <https://github.com/antlr/grammars-v4/blob/master/python/python3/Python3Lexer.g4> .

STRING_LITERAL
    : ( [rR] | [fF] | ( [fF] [rR] ) | ( [rR] [fF] ) )? ( SHORT_STRING | LONG_STRING )
    ;

fragment SHORT_STRING
    : '\'' ( STRING_ESCAPE_SEQ | ~[\\\r\n\f'] )* '\''
    | '"' ( STRING_ESCAPE_SEQ | ~[\\\r\n\f"] )* '"'
    ;

fragment LONG_STRING
    : '\'\'\'' ( STRING_ESCAPE_SEQ | ~'\\' )*? '\'\'\''
    | '"""' ( STRING_ESCAPE_SEQ | ~'\\' )*? '"""'
    ;

fragment STRING_ESCAPE_SEQ
    : '\\' .
    ;

/// Boolean literals.
BOOLEAN
    : TRUE
    | FALSE
    ;

IF: 'if';
ELSE: 'else';
ELIF: 'elif';
FOR: 'for';
WHILE: 'while';
BREAK: 'break';
CONTINUE: 'continue';
DO: 'do';
DEF: 'def';
RETURN: 'return';
YIELD: 'yield';
CLASS: 'class';
TRY: 'try';
EXCEPT: 'except';
FINALLY: 'finally';
WITH: 'with';
ASSERT: 'assert';
IMPORT: 'import';
FROM: 'from';
AS: 'as';
AND: 'and';
OR: 'or';
NOT: 'not';
IN: 'in';
IS: 'is';
TRUE : 'true';
FALSE : 'false';
NIL: 'nil';

DOT: '.';
COMMA: ',';
SEMICOLON: ';';
AT: '@';
CRLF: '\r'? '\n';

PLUS : '+';
MINUS : '-';
MUL : '*';
DIV : '/';
MOD : '%';
EXP : '**';

EQUAL : '==';
NOT_EQUAL : '!=';
GREATER : '>';
LESS : '<';
LESS_EQUAL : '<=';
GREATER_EQUAL : '>=';

ASSIGN : '=';
PLUS_ASSIGN : '+=';
MINUS_ASSIGN : '-=';
MUL_ASSIGN : '*=';
DIV_ASSIGN : '/=';
MOD_ASSIGN : '%=';
EXP_ASSIGN : '**=';

BIT_AND : '&';
BIT_OR : '|';
BIT_XOR : '^';
BIT_NOT : '~';
BIT_SHL : '<<';
BIT_SHR : '>>';

LEFT_PAREN : '(';
RIGHT_PAREN : ')';
LEFT_BRACK : '[';
RIGHT_BRACK : ']';
LEFT_BRACE: '{';
RIGHT_BRACE: '}';

/// Identifiers.

ID
    : ID_START ID_CONTINUE*
    ;

// TODO: See https://github.com/antlr/grammars-v4/blob/master/python/python3/Python3Lexer.g4#L383 for unicode support.
fragment ID_START
    : '_'
    | [a-zA-Z]
    | '$'
    ;

fragment ID_CONTINUE
    : ID_START
    | [0-9]
    ;

WS : [ \t]+ -> skip ;
COMMENT: '#' ~[\r\n\f]* -> skip;
