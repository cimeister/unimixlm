
import GraphicDevice from "./GraphicDevice"
import Transform from "../Transform"
import Light from "../Light/Light"
import IModel from "./IModel"
import IShader from "../Resource/IShader"
import IMaterial from "../Resource/IMaterial"
import IGeometry from "../Resource/IGeometry"
import IMesh from "../Resource/IMesh"
import Matrix4x4 from "../Math/Matrix"
import Vector3 from "../Math/Vector3"



class ExParameter{
    slot:number;
    param;
    size:number;
    atachFunc:Function;
    constructor(arg_slot:number,arg_size:number,arg_param){
        this.slot=arg_slot;
        this.param=arg_param;
        this.size=arg_size;
        
        switch(arg_size){
            case 1:
            this.atachFunc=this.Atach1f;
            break;
            case 2:
            this.atachFunc=this.Atach2f;
            break;
            case 3:
            this.atachFunc=this.Atach3f;
            break;
            case 4:
            this.atachFunc=this.Atach4f;
            break;
            case 16:
            this.atachFunc=this.AtachMat;
            break;
        }
    }


    Atach1f(device:GraphicDevice){
        device.context.uniform1fv(device.shader.uniformLocations[this.slot], this.param.data);
    }
    Atach2f(device:GraphicDevice){
        device.context.uniform2fv(device.shader.uniformLocations[this.slot], this.param.data);
    }
    Atach3f(device:GraphicDevice){
        device.context.uniform3fv(device.shader.uniformLocations[this.slot], this.param.data);
    }
    Atach4f(device:GraphicDevice){
        device.context.uniform4fv(device.shader.uniformLocations[this.slot], this.param.data);
    }
    AtachMat(device:GraphicDevice){
        device.context.uniformMatrix4fv(device.shader.uniformLocations[this.slot],false, this.param.data);
    }
}

export default class Model implements IModel{
    graphicDevice:GraphicDevice;
    geometry:IGeometry;
    materials:Array< IMaterial>;
    shader:IShader;
    transform:Transform;
    lights:Array<Light>;
    subsets:Array<number>;
    exParams:Array<ExParameter>;
    
    Draw:Function;
    constructor (isLight:boolean ,isBillBoard?:boolean){
        if(isBillBoard==true){
        if(isLight)
        this.Draw=this.Draw_BillBoard_Light;
        else
        this.Draw=this.Draw_BillBoard_NonLight;
        }else{

            if(isLight)
            this.Draw=this.Draw_Light;
            else
            this.Draw=this.Draw_NonLight;
        }
        this.exParams=new Array();
    }
    Initialize_geom(arg_graphicDevice:GraphicDevice,arg_geometry:IGeometry,arg_material:IMaterial,arg_Shader:IShader,arg_transform:Transform){
        this.graphicDevice=arg_graphicDevice;
        this.geometry=arg_geometry;
        this.materials=new Array();
        this.materials.push( arg_material);
        this.subsets=new Array();
        this.subsets.push(this.geometry.GetIndexSize());
        this.shader=arg_Shader;
        this.transform=arg_transform;
        this.lights=new Array();
    }
    Initialize_mesh(arg_graphicDevice:GraphicDevice,arg_mesh:IMesh,arg_Shader:IShader,arg_transform:Transform){
        this.graphicDevice=arg_graphicDevice;
        this.geometry=arg_mesh.geometry;
        this.materials=arg_mesh.ary_materials;
        this.subsets=this.geometry.GetSubSet();
        
        this.shader=arg_Shader;
        this.transform=arg_transform;
        this.lights=new Array();
    }
    SetLight(arg_light:Light):void{
        this.lights.push(arg_light);
    }

    AddExParam(arg_slot:number,arg_size:number,arg_param){
        this.exParams.push(new ExParameter(arg_slot,arg_size,arg_param));
    }
    Draw_NonLight():void{

        this.shader.Attach();
        var mvpMatrix=this.graphicDevice.TempMatrix.Multiply( this.transform.Matrix );
        var invMatrix=this.transform.Matrix.Inverse();
        
        // uniform変数の登録と描画
        
        this.graphicDevice.context.uniformMatrix4fv(this.shader.uniformLocations[0], false, mvpMatrix.data);
        this.graphicDevice.context.uniformMatrix4fv(this.shader.uniformLocations[1], false, this.transform.Matrix.data);
        this.graphicDevice.context.uniformMatrix4fv(this.shader.uniformLocations[2], false, invMatrix.data);

        
        
        for(var i=0;i< this.exParams.length;i++){
            this.exParams[i].atachFunc(this.graphicDevice);
        }
        
        this.geometry.Draw();
        var offset=0;
        
        for(var i=0;i< this.subsets.length;i++){
            this.materials[i].Attach();
            this.graphicDevice.context.drawElements(this.graphicDevice.context.TRIANGLES,this.subsets[i] , this.graphicDevice.context.UNSIGNED_SHORT, offset*2);
            offset+=this.subsets[i];

        }
    }
    Draw_Light():void{

        this.shader.Attach();
        var mvpMatrix=this.graphicDevice.TempMatrix.Multiply( this.transform.Matrix );
        var invMatrix=this.transform.Matrix.Inverse();
        
        // uniform変数の登録と描画
        
        this.graphicDevice.context.uniformMatrix4fv(this.shader.uniformLocations[0], false, mvpMatrix.data);
        this.graphicDevice.context.uniformMatrix4fv(this.shader.uniformLocations[1], false, this.transform.Matrix.data);
        this.graphicDevice.context.uniformMatrix4fv(this.shader.uniformLocations[2], false, invMatrix.data);

        
        this.lights.forEach(light=>light.Atach());
        
        for(var i=0;i< this.exParams.length;i++){
            this.exParams[i].atachFunc(this.graphicDevice);
        }
        this.geometry.Draw();
        var offset=0;
        
        for(var i=0;i< this.subsets.length;i++){
            this.materials[i].Attach();
            this.graphicDevice.context.drawElements(this.graphicDevice.context.TRIANGLES,this.subsets[i] , this.graphicDevice.context.UNSIGNED_SHORT, offset*2);
            offset+=this.subsets[i];

        }
    }
    Draw_BillBoard_Light():void{
        
        this.shader.Attach();

        var mMatrix=this.transform.Matrix.Multiply( this.graphicDevice.CameraRotationInv);

        var mvpMatrix=this.graphicDevice.TempMatrix.Multiply(  mMatrix);
        var invMatrix=this.transform.Matrix.Inverse();
        
        this.lights.forEach(light=>light.Atach());
        // uniform変数の登録と描画
        
        this.graphicDevice.context.uniformMatrix4fv(this.shader.uniformLocations[0], false, mvpMatrix.data);
        this.graphicDevice.context.uniformMatrix4fv(this.shader.uniformLocations[1], false, this.transform.Matrix.data);
        this.graphicDevice.context.uniformMatrix4fv(this.shader.uniformLocations[2], false, invMatrix.data);

        
        
        for(var i=0;i< this.exParams.length;i++){
            this.exParams[i].atachFunc(this.graphicDevice);
        }
        
        this.geometry.Draw();
        var offset=0;
        
        for(var i=0;i< this.subsets.length;i++){
            this.materials[i].Attach();
            this.graphicDevice.context.drawElements(this.graphicDevice.context.TRIANGLES,this.subsets[i] , this.graphicDevice.context.UNSIGNED_SHORT, offset*2);
            offset+=this.subsets[i];

        }
    }
    Draw_BillBoard_NonLight():void{
        
        this.shader.Attach();
        //this.transform.LookAt(this.graphicDevice.CameraPosition,new Vector3(0,1,0));
        var mMatrix=this.transform.Matrix.Multiply( this.graphicDevice.CameraRotationInv);

        
        var mvpMatrix=this.graphicDevice.TempMatrix.Multiply(  mMatrix);
        var invMatrix=this.transform.Matrix.Inverse();
        
        // uniform変数の登録と描画
        
        this.graphicDevice.context.uniformMatrix4fv(this.shader.uniformLocations[0], false, mvpMatrix.data);
        this.graphicDevice.context.uniformMatrix4fv(this.shader.uniformLocations[1], false, this.transform.Matrix.data);
        this.graphicDevice.context.uniformMatrix4fv(this.shader.uniformLocations[2], false, invMatrix.data);

        
        
        for(var i=0;i< this.exParams.length;i++){
            this.exParams[i].atachFunc(this.graphicDevice);
        }
        
        this.geometry.Draw();
        var offset=0;
        
        for(var i=0;i< this.subsets.length;i++){
            this.materials[i].Attach();
            this.graphicDevice.context.drawElements(this.graphicDevice.context.TRIANGLES,this.subsets[i] , this.graphicDevice.context.UNSIGNED_SHORT, offset*2);
            offset+=this.subsets[i];

        }

    }

    
}