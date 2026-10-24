<?php
require_once ("conexion.php");
$conexion = conectarBD();
$usuario = $_GET['user'];
$contrasena = $_GET['pass'];

require_once ("buscarAdministrador.php");
$idAdministrador = buscarAdministrador( $usuario, $contrasena, $conexion);
$idCliente = $_GET['idCliente'];
$nombre = $_GET['nombre'];
$apellidoPaterno = $_GET['apellidoPaterno'];
$apellidoMaterno = $_GET['apellidoMaterno'];
$telefono = $_GET['telefono'];
$domicilio = $_GET['domicilio'];
$diaIngreso = $_GET['diaIngreso'];

$ok = modificarCliente( $conexion, $idCliente, $nombre, $apellidoPaterno, $apellidoMaterno, $telefono, $domicilio, $diaIngreso, $idAdministrador );
$opcion = -1;

if($ok != false)   {
    $opcion = 2;
}

echo '<form id="elForm" action="http://localhost/Proyecto%20modular/adminCliente.php">
        <input type="hidden" name="user" value='.$usuario.'>
        <input type="hidden" name="pass" value='.$con<PASSWORD>.'>
        <input type="hidden" name="opcion" value='.$opcion.'>
        <input type="hidden" name="etiquetaModificar2" value=true>
        </form>';
echo '<script type="text/javascript">
    document.forms["elForm"].submit()
    </script>';
    

    function modificarCliente( $conexion, $idCliente, $nombre, $apellidoPaterno, $apellidoMaterno, $telefono, $domicilio, $diaIngreso, $idAdministrador )
    {
        $sql = "UPDATE cliente SET nombre='".$nombre."', apellidopaterno='".$apellidoPaterno."', apellidomaterno='".$apellidoMaterno."', telefono='".$telefono."', domicilio='".$domicilio."', diaingreso='".$diaIngreso."', idadministrador='".$idAdministrador."' WHERE idcliente=".$idCliente;
        return pg_query( $conexion, $sql );
    }
?>