<?php

Use Mini\Core\Session;

$this->layout('layout');
Session::init();
?>

<form action="<?= URL; ?>productos/inserta" method="post" enctype="multipart/form-data">
    <div class="banner">
        <h4>Ingresa producto <?php echo Session::get('Nick') ?> </h4>

        <div class="login">
            <br>
            <label for="nombre">Nombre del Producto</label>
            <?php if (isset($mensaje)) {
                ?>
                <div>
                    <h3 style="color:green"><?php echo $mensaje; ?></h3>
                </div>
            <?php } ?>
            <input type="text" class="form-control" name="nombre" placeholder="Nombre"
                <?php if (isset($datos['nombre']))
                    echo ' value="' . $datos['nombre'] . '"'; ?>>
            <?php if (isset ($errors['nombre']))
                echo '<span class="errorf">' . $errors['nombre'] . '</span><br>';
            ?>

            <label for="modelo">Modelo del Producto</label>
            <input type="text" class="form-control" id="modelo" name="modelo" placeholder="Modelo"
                <?php if (isset($datos['modelo']))
                    echo ' value="' . $datos['modelo'] . '"'; ?>>
            <?php if (isset ($errors['modelo']))
                echo '<span class="errorf">' . $errors['modelo'] . '</span><br>';
            ?>

            <label for="marca">Marca del producto</label>
            <input type="text" class="form-control" id="marca" name="marca" placeholder="Marca"
                <?php if (isset($datos['marca']))
                    echo ' value="' . $datos['marca'] . '"'; ?>>
            <?php if (isset ($errors['marca']))
                echo '<span class="errorf">' . $errors['marca'] . '</span><br>';
            ?>

            <label for="categoria">Categorias</label>
            <select class="form-control" name="categoria">
                <?php if (count($categorias) == 0) :
                    $categoria = 'No existen categorias';
                    echo '<option>' . " $categoria " . '</option>'; ?>
                <?php else : ?>
                    <?php foreach ($categorias as $categoria):
                        $categoria = $categoria->nombre;
                        echo '<option>' . " $categoria " . '</option>';
                        ?>
                    <?php endforeach; ?>
                <?php endif; ?>
            </select>

            <label for="descripcion">Descripción</label>
            <textarea class="form-control" name="descripcion" rows="2"
            ><?php if (isset($datos['descripcion'])) {
                    echo $datos['descripcion'];
                } ?>
            </textarea>
            <?php if (isset ($errors['descripcion']))
                echo '<span class="errorf">' . $errors['descripcion'] . '</span><br>';
            ?>

            <label for="precio">Precio</label>
            <input style="width:100px; margin:auto;" class="form-control" id="precio" name="precio" type="number"
                   step="any"
            <?php if (isset($datos['precio'])) {
                echo ' value="' . $datos['precio'] . '"';
            } ?>>
            <?php if (isset ($errors['precio']))
                echo '<span class="errorf">' . $errors['precio'] . '</span><br>';
            ?>

            <label for="imagen">Imagen</label>
            <input type="file" name="imagen" class="form-control" height="40px"/>
            <br><br>

            <button type="submit" class="btn" name="addproductos">Enviar</button>
            <br><br>
        </div>
    </div>
</form>