package com.example;

import de.greenrobot.daogenerator.DaoGenerator;
import de.greenrobot.daogenerator.Entity;
import de.greenrobot.daogenerator.Schema;

public class ThreadInfoDaoGeneater {

    public static final String DAO_PATH = "C:/Users/LYW/AndroidStudioProjects/D" +
            "owload_File_App/app/src/main/java-gen";
    public static final String PACKAGE_NAME = "xiaomeng.bupt.com.donload_file_app.greendao";

    public static void main(String[] args)throws Exception{
        Schema schema = new Schema(1,PACKAGE_NAME);
        addThreadInfo(schema);
        new DaoGenerator().generateAll(schema,DAO_PATH);
    }

    private static void addThreadInfo(Schema schema) {
        //表名为ThreadInfo
        Entity threadInfo = schema.addEntity("ThreadInfo");
        threadInfo.addIdProperty().primaryKey().autoincrement();
        threadInfo.addIntProperty("start");
        threadInfo.addIntProperty("stop");
        threadInfo.addIntProperty("finished");
        threadInfo.addStringProperty("url");
    }
}