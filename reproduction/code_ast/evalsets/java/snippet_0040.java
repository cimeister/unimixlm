package es.cic.curso.grupo3.ejercicio027.service;

import java.util.List;

import es.cic.curso.grupo3.ejercicio027.domain.Tipo;

public interface TipoService {

	Long agregarTipo(String nombre);

	Tipo agregarTipo(Tipo tipo);

	Tipo actualizarTipo(Tipo modificado);

	Tipo obtenerTipo(Long id);

	List<Tipo> obtenerListaTipos();

	void eliminarTipo(Long id);

	Tipo obtenerTipoPorNombre(String nombre);

}