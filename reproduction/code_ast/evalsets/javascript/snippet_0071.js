/* HttpServer.js - Http server class
 *
 * Copyright (C) 2016 <NAME>
 *
 * This software may be modified and distributed under the terms
 * of the MIT license.  See the LICENSE file for details.
 */
import {createServer} from "http";
import {EventEmitter} from "events";
import {parse as parseQuerystring} from "querystring";
import {extname} from "path";
import {parse as parseUrl} from "url";
import Logger from "core/Logger";

const log = new Logger( "HttpServer" );

/**
 * @class HttpServer
 */
class HttpServer extends EventEmitter {
    constructor() {
        super();
        this._listening = false;
        this._listener = null;
    }

    /**
     * Adds the listener function to the end of the listeners array for the specified event
     * @param  {string} event
     * @param  {Function} listener
     * @return {EventEmitter} Current instance
     */
    on( event: string, listener: Function ): HttpServer {
        if ( this.listeners( event ) > 0 ) {
            throw Error( "Cannot add more than 1 listener for event" );
        }
        
        return super.on( event, listener );
    }

    /**
     * Binds listener to given port
     * @param  {Number} port - Port where to listen for http requests
     * @return {HttpServer} Current instance
     */
    listen( port: number ): HttpServer {
        if ( this._listening ) {
            throw Error( "Server is already listening" );
        }

        this._listener = createServer( processRequest.bind( this ) );
        this._listener.listen( port );
        this._listening = true;
        log.info( `Listening on port ${port}` );
        return this;
    }
}

/**
 * Processes http request
 * @param  {http.IncomingMessage} request
 * @param  {http.ServerResponse} response
 * @return {void}
 */
function processRequest( request: Object, response: Object ): void {
    log.info( `request to ${request.url}` );

    readData( request, ( error, data ) => {
        var event = "";
        var url = parseUrl( request.url );

        if ( error ) {
            event = "error";
            this.emit( event, url, error, callback );
        } else if ( request.method === "GET" && extname( request.url ) ) {
            event = "file";
            this.emit( event, url, callback );
        } else {
            event = "request";
            this.emit( event, url, data, callback );
        }

        if ( this.listenerCount( event ) === 0 ) {
            log.fatal( `no event listeners for ${event} event. aborting...` );
            response.writeHead( 400 );
            response.end();
            return;
        }

        // Body parameter must be always buffer in order to be monomorphic function
        function callback( status: number, headers: Object, body: Buffer ): void {
            response.writeHead( status, headers );
            response.end( body );
            log.info( `response for ${request.url} with: ${body}` );
        }
    } );
}

/**
 * Reads data from client request if exists
 * @param  {http.IncomingMessage} request
 * @param  {Function} callback
 * @return {void}
 */
function readData( request: Object, callback: Function ): void {
    var dataString = "";

    if ( request.method !== "POST" && request.method !== "PUT" ) {
        callback( null, null );
        return;
    }

    request.on( "data", buffer => dataString += buffer.toString() );
    request.on( "end", function processData() {
        var data = null;
        var error = null;
        try {
            data = parseData( dataString, request.headers[ "content-type" ] );
        } catch ( err ) {
            error = err;
        } finally {
            callback( error, data );
        }
    } );
}

/**
 * Parses data received from client
 * @param  {string} dataString
 * @param  {Number} type
 * @return {Object}
 */
function parseData( dataString: string, contentType: string ): Object {
    switch ( contentType ) {
        case "application/json": return JSON.parse( dataString );
        case "x-form-urlencoded": return parseQuerystring( dataString );
        default: throw Error( "Unknown data type" );
    }
}

export default HttpServer;