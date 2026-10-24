var Evy = function( program ) {

  this.log = new Log( "Evy", Log.Level.DEBUG );
  this.log.trace( this, "[ctor](", program, ")" );

  this.root = new Tree( program );
  this.events = cloneOf( this.events );

  applyNativeSubscriptions( this );

};

Evy.prototype = {

  events: {},

  subscribe: function( event, behavior ) {

    this.log.trace( this, "subscribe(", event, behavior, ")" );

    if( !event ) { throw "Event name must be specified."; }
    if( !behavior ) { throw "Behavior must be a function."; }
    if( !this.events[event] ) { this.events[event] = []; }

    this.events[event].push( behavior );

  },

  unsubscribe: function( event, behavior ) {

    this.log.trace( this, "unsubscribe(", event, behavior, ")" );

    if( !event ) { throw "Event name must be specified."; }
    if( !behavior ) { this.events[event] = []; return; }

    var after = [];
    toCollection( this.events[event] ).each( function() {
      if( this !== behavior ) {
        after.push( this );
      }
    } );

    this.events[event] = after;

  },

  publish: function( event, parameters, context ) {

    this.log.trace( this, "publish(", event, parameters, context, ")" );

    if( !event ) { throw "Event name must be specified."; }
    if( !parameters ) { parameters = []; }
    if( !context ) { context = {}; }

    toCollection( this.events[event] ).each( function() {
      this.apply( context, parameters );
    } );

  },

  execute: ( function() {
  
    var PUBLICATION = /^([^ ]+)(?: (.+))?$/gi;
    var IDENTIFIER = /^[^\d"\.\+\-'\s][^"\.\+\-'\s]*/gi;

    function isSymbol( value ) {
      return !!( value || "" ).match( IDENTIFIER );
    }

    function expandSymbols( context, parameters ) {
      for( var k = 0; k < parameters.length; k++ ) {
        var parameter = parameters[k];
        if( parameter.length == 2 && parameter[1] !== undefined ) {
          var key = parameter[0];
          var value = parameter[1];
          context.symbols[key] = isSymbol(value) ? context.lookup(value) : eval(value);
        } else if( isSymbol(parameter[0]) ) {
          parameter[1] = context.lookup(parameter[0]);
        } else {
          parameter[0] = eval(parameter[0]);
        }
      }
    }

    return function( context ) {
  
      this.log.trace( this, "execute(", context, ")" );

      if( !context ) { context = this.root; }

      if( context.name ) {

        var event = context.name.replace( PUBLICATION, "$1" );
        var parameters = parse( context.name.replace( PUBLICATION, "$2" ) );

        expandSymbols( context, parameters );

        this.publish( event, parameters, context );

      } else {

        for( var i = 0; i < context.children.length; i++ ) {
          this.execute( context.children[i] );
        }

      }

    };

  } )()

};