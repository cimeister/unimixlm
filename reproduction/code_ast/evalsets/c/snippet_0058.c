#include <stdlib.h>
#include <stack.h>

STACK *stack_create() {
    STACK *stack;

    stack = (STACK *) malloc(sizeof(STACK));

    if (stack == NULL) {
        return NULL;
    }

    stack->head = NULL;
    stack->tail = NULL;
    stack->size = 0;

    return stack;
}

unsigned int stack_push(STACK *stack, const STACK_DATA data) {
    STACK_NODE *new_node;

    new_node = (STACK_NODE *) malloc(sizeof(STACK_NODE));

    if (new_node == NULL) {
        return 0;
    }

    new_node->data = data;

    if(stack->size == 0) {
        new_node->next = stack->head;
        stack->head = new_node;
    }
    else {
        new_node->next = stack->tail;
    }

    stack->tail = new_node;

    stack->size++;

    return stack->size;
}

STACK_DATA stack_pop(STACK *stack) {
    if(stack->size == 0) {
        return NULL;
    }

    STACK_NODE *last = stack->tail;
    STACK_DATA data = last->data;

    stack->tail = last->next;

    if(stack->tail == NULL) {
        stack->head = NULL;
    }

    free(last);
    stack->size--;

    return data;
}

int stack_size(const STACK *stack) {
    return stack->size;
}

int stack_is_empty(const STACK *stack) {
    return (stack->size == 0 ? 1 : 0);
}

void stack_destroy(STACK **ref_stack) {
    if((*ref_stack)->size == 0) {
        free(*ref_stack);
        *ref_stack = NULL;

        return;
    }

    STACK_NODE *current;
    current = (*ref_stack)->tail;

    while(current != NULL) {
        STACK_NODE *tmp = current->next;
        free(current);
        current = tmp;
    }

    free(*ref_stack);
    *ref_stack = NULL;
}

STACK_API *new_stack_api() {
    STACK_API *_stack_api;
    _stack_api = NULL;

    _stack_api = (STACK_API *) malloc(sizeof(STACK_API));

    if (_stack_api == NULL) {
        //exit(MEMORY_ALLOC_ERROR);
        exit(-1);
    }

    _stack_api->create_stack = &(stack_create);
    _stack_api->push = &(stack_push);
    _stack_api->pop = &(stack_pop);
    _stack_api->size = &(stack_size);
    _stack_api->is_empty = &(stack_is_empty);
    _stack_api->destroy = &(stack_destroy);

    return _stack_api;
}