#include "mp4.h"
#include "ui_mp4.h"

Mp4::Mp4(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::Mp4)
{
    ui->setupUi(this);

    connect(ui->buttonBox, SIGNAL(), this, SLOT(handle_entrance()));
}

Mp4::~Mp4()
{
    delete ui;
}

//void Mp4::on_buttonBox_accepted()
//{

//}

//void Mp4::on_buttonBox_clicked(QAbstractButton *button)
//{
////    accept();
//    if (button->isEnabled())
//        button->setEnabled(false);
//}

void Mp4::handle_entrance()
{
    accept();
}