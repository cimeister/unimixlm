'use strict';
var fs = require('fs');
const configFile = require('../../../../config/rights.manager/menu.rights.manager/menu.rights.manager.config');


exports.getMenuRightsConfig = function(req, res) {
    const configData = configFile.menurights_columns;
    return res.status(200).json({
        success: true,
        configData: configData
    })
};

exports.changeMenuRightsConfig = function(req, res) {
    const params = req.query;
    var content = `
        "use strict";
        module.exports = {
            menurights_columns:[
                {title:${JSON.stringify(params.ROLE_NAME)},dataIndex:'ROLE_NAME'},
            ]
        }`;
    fs.writeFile('server/config/rights.manager/menu.rights.manager/menu.rights.manager.config.js', content,function(err){
        if(err){
            console.error(err);  
        }else{
            console.log('写文件操作成功');
        } 
        
    });
    
    return res.status(200).json({
        success: true,
      
    })
};