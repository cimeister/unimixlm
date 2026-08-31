package com.adportas.videollamadas.service.interfaces;

import com.adportas.videollamadas.domain.ContactoAgente;
import com.adportas.videollamadas.websocket.SesionWebsocket;
import java.io.IOException;
import org.springframework.web.socket.WebSocketSession;

/**
 *
 * @author benjamin
 * @param <T>
 */
public interface IWebsocketService<T> {

    void addSession(WebSocketSession session);
    
    void registerUser(String sessionId,ContactoAgente contactoAgente);
    
    SesionWebsocket findSesionWebsocketBySeessionId(String sessionId);

    void removeSession(WebSocketSession session);

    void sendMessage(ContactoAgente contactoAgente,T mensaje) throws IOException;

    void sendBroadcastMessage(T mensaje) throws IOException;
}