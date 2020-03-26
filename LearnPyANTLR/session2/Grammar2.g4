grammar Grammar2;

// Fragment & Lexer

DO: 'do';
DEF: 'def';
END: 'end';
ID: [_a-zA-Z][_0-9a-zA-Z]*;
DECIMAL_INTEGER: [1-9][0-9]* | '0';

ASSIGN: '=';
COMMA: ',';
DOLLAR: '$';
DOT: '.';
LB: '{';
LP: '(';
LSP: '[';
MINUS: '-';
MULT: '*';
PLUS: '+';
RB: '}';
RP: ')';
RSP: ']';

NL: '\r'? '\n' | ';';
WS : [ \t]+ -> skip ;

// Parser
prog
    : exprList EOF
    ;

exprList
    : NL* (expr NL+)* expr?
    ;

expr
    : atom                                      # atomExpr
    | expr DOT expr                             # getAttrExpr
    | assignLHS ASSIGN expr                     # assignExpr
    | expr doBlock                              # funcCallExpr
    | expr LP funcCallParams RP doBlock?        # funcCallExpr
    | expr (PLUS | MINUS) expr                  # arithExpr
    | LP expr RP                                # parenExpr
    | funcDef                                   # funcDefExpr
    ;

assignLHS
    : assignLHSItem (COMMA assignLHSItem)* COMMA?
    ;

assignLHSItem
    : '*'? lValue
    | LP assignLHS RP
    ;

funcCallParams
    :
    | (funcCallParam COMMA)* funcCallParam COMMA?
    ;

funcCallParam
    : expr
    | ID ASSIGN expr
    ;

doBlock
    : DO (funcDefArgs NL)? exprList END
    ;

funcDef
    : DEF ID LP funcDefArgs RP exprList END
    ;

funcDefArgs
    :
    | (funcDefArg COMMA)* funcDefArg COMMA?
    ;

funcDefArg
    : ID
    ;

atom
    : lValue
    | integer
    ;

lValue
    : ID
    | DOLLAR
    | lValue DOT ID
    | lValue LSP expr RSP
    ;

integer
    : DECIMAL_INTEGER
    ;
