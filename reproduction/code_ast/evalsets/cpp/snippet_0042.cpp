#include "minesweeperwelcomeview.h"
#include "ui_minesweeperwelcomeview.h"

#include <QPushButton>

namespace MS {
MinesweeperWelcomeView::MinesweeperWelcomeView(QWidget* parent)
    : QWidget(parent)
    , ui_(new Ui::MinesweeperWelcomeView)
{
    ui_->setupUi(this);
    connect(ui_->quitPushButton, &QPushButton::clicked, this,
            &MinesweeperWelcomeView::quit);
    connect(ui_->playPushButton, &QPushButton::clicked, [this] {
        auto checkedButton = ui_->buttonGroup->checkedButton();
        if (!checkedButton) {
            return;
        }
        auto targetString = checkedButton->text();
        if (targetString == ui_->easyRadioButton->text()) {
            emit start(8, 8, 10);
        }
        else if (targetString == ui_->middleRadioButton->text()) {
            emit start(16, 16, 40);
        }
        else if (targetString == ui_->hardRadioButton->text()) {
            emit start(16, 30, 99);
        }
        else if (targetString == ui_->customRadioButton->text()) {
            bool rowOk = false;
            auto row = ui_->rowLineEdit->text().toInt(&rowOk);
            if (!rowOk || row <= 0) {
                return;
            }
            bool columnOk = false;
            auto column = ui_->columnLineEdit->text().toInt(&columnOk);
            if (!columnOk || column <= 0) {
                return;
            }
            bool mineOk = false;
            auto mine = ui_->minesLineEdit->text().toInt(&mineOk);
            if (!mineOk || !(0 < mine && mine < (row * column))) {
                return;
            }
            emit start(row, column, mine);
        }
        hide();
    });
}

MinesweeperWelcomeView::~MinesweeperWelcomeView()
{
}
} // namespace MS