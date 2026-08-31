#include "framework.h"
#include "graphics/rendersubsystem.h"
#include "graphics/renderqueue.h"
#include "graphics/renderqueuepriority.h"

NAMESPACE_BEGIN(sway)

/*!
 * \brief
 *   Конструктор класса.
 *
 *   Выполняет инициализацию нового экземпляра класса.
 */
Framework::Framework() {
	_renderSubsystem = boost::make_shared<graphics::RenderSubsystem>(this);
	registerObject(_renderSubsystem.get());

	_renderQueue = _renderSubsystem->createQueue();
	_renderQueue->setPriority(graphics::kRenderQueuePriority_Normal);

	_finiteStateMachine = new game::GameStateMachine();
}

/*!
 * \brief
 *   Деструктор класса.
 *
 *   Освобождает захваченные ресурсы.
 */
Framework::~Framework() {
	unregisterObject(_renderSubsystem->getClassName());

	delete _finiteStateMachine;
}

void Framework::start(game::GameState *state) {
	_finiteStateMachine->changeState(state, this);
}

void Framework::update() {
	_finiteStateMachine->update();

	math::TColor<f32> clr = math::TColor<f32>(0, 0.5, 1, 1);
	glClearColor(clr.getR(), clr.getG(), clr.getB(), clr.getA());
	glClear(GL_COLOR_BUFFER_BIT);

	_renderSubsystem->render();
}

NAMESPACE_END(sway)