import { TraitDoublyLinkedListNodeGetPrevious } from './traits/trait-doubly-linked-list-node-get-previous/trait-doubly-linked-list-node-get-previous';
import { TraitDoublyLinkedListNodeSetPrevious } from './traits/trait-doubly-linked-list-node-set-previous/trait-doubly-linked-list-node-set-previous';
import { ISingleLinkedListNodeLike } from '../../single/node/single-linked-list-node-interface';

export interface IDoublyLinkedListNodeLike<GSelf extends IDoublyLinkedListNodeLike<any>> extends ISingleLinkedListNodeLike<GSelf>,
  // traits
  TraitDoublyLinkedListNodeGetPrevious<GSelf>,
  TraitDoublyLinkedListNodeSetPrevious<GSelf>
  //
{
}