grammar simple;

// Fragment & Lexer
HELLO: [Hh] 'ello';
WORLD: [Ww] 'orld';
COMMA: ',';
WS : [ \t\r\n]+ -> skip ;

// Parser
main: HELLO COMMA? WORLD '!'?;
