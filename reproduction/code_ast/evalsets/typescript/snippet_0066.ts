
/*------------------------------------------------------------------*
 * user-interactions.ts												*
 *																	*
 * Historico:	07/12/2021	Primer version							*
 *																	*
 * Funciones que permiten la interaccion del usuario con la aplica- *
 * cion.                                                            *
 *------------------------------------------------------------------*/


  /*------------------------------------------------------------------*
  * userClickDelete(id)							                     *
  *																	 *
  * Entrada:     id          Indice del dispositivo a borrar dentro  *
  *                          del arreglo local de dispositivos.      *
  *																	 *
  * Salida:		 No posee                                            *
  *																	 *
  * Retorno:	 No posee                                            *
  *																	 *
  * Descripcion:													 *
  *																	 *
  * Solicita la eliminacion de un dispositivo                        *
  *------------------------------------------------------------------*/

function userClickDelete(id : number)
{   
    let f = userClickDelete.name;

    logger(f, `User wants to delete device ${id}`)
    
    deletetDevice(id, (success : boolean) => 
        {
            // No se borro del remoto
            if(success === false)
                return;
            // Se borro del remoto, borrarlo del local
            for(let i = 0 ; i < devArrayGlobal.length; i++) {
                if(id === devArrayGlobal[i].id) {
                    logger(f, `Element ${i} removed from global device array. Redrawing device list.`);
                    devArrayGlobal.splice(i, 1);
                    drawDevices();
                }
            }
        }
    )
}

  /*------------------------------------------------------------------*
  * userClickAdd()							                         *
  *																	 *
  * Entrada:     No posee                                            *
  *																	 *
  * Salida:		 No posee                                            *
  *																	 *
  * Retorno:	 No posee                                            *
  *																	 *
  * Descripcion:													 *
  *																	 *
  * Carga la pagina para agregar un dispositivo                      *
  *------------------------------------------------------------------*/

function userClickAdd()
{
    let f = userClickAdd.name;
    logger(f, `User wants to add device`);
    showAppPage2();
}

  /*------------------------------------------------------------------*
  * userClickAdd(id)							                     *
  *																	 *
  * Entrada:     id          Indice del dispositivo a editar dentro  *
  *                          del arreglo local de dispositivos.      *
  *																	 *
  * Salida:		 No posee                                            *
  *																	 *
  * Retorno:	 No posee                                            *
  *																	 *
  * Descripcion:													 *
  *																	 *
  * Carga la pagina para editar un dispositivo                       *
  *------------------------------------------------------------------*/

function userClickEdit(i : number)
{
    let f = userClickEdit.name;
    logger(f, `User wants to edit device[${i}] `);
    showAppPage2(i);
}