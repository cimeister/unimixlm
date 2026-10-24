package gui;

import javax.swing.*;

public class RunFormDataMahasiswa {
    public static void main(String[] args) {
        JFrame jFrame = new JFrame("Aplikasi Mahasiswa FST USR");
        jFrame.setContentPane(new FormDataMahasiswa().getLootPanel());
        jFrame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        jFrame.setLocationRelativeTo(null);
        jFrame.setSize(400, 300);
        jFrame.setVisible(true);
    }
}