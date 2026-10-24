#pragma once

#include "../Widget/ProcessorBaseWidget.h"
#include "BaseProcessor.h"

QT_BEGIN_NAMESPACE
class QLineEdit;
QT_END_NAMESPACE

class EqualizationWidget : public ProcessorBaseWidget
{
    Q_OBJECT

public:
    EqualizationWidget(BaseProcessor* processor, QWidget* parent = nullptr);

private slots:
    void editChanged();

signals:
    void valueChanged(int gridSize, int clipLimit);

private:
    QLineEdit* _gridSizeEdit;
    QLineEdit* _clipLimitEdit;
};

class EqualizationProcessor : public BaseProcessor
{
    Q_OBJECT

public:
    EqualizationProcessor(QObject* parent = nullptr);
    virtual ~EqualizationProcessor();

    void initUI() override;

public slots:
    void valueChanged(int gridSize, int clipLimit);

protected:
    void processImage(GeneralImage* srcImage, GeneralImage* dstImage) override;

    void processImage(MonoImage* srcImage, MonoImage* dstImage) override;

private:
    void RGB2HSV(uchar R, uchar G, uchar B, int& H, float& S, uchar& V);
    void HSV2RGB(int H, float S, uchar V, uchar& R, uchar& G, uchar& B);

private:
    EqualizationWidget* _widget;
    int _gridSize;
    int _clipLimit;
};