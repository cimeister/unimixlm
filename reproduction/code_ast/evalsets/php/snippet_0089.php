<?php
/**
 * Created by PhpStorm.
 * User: junior
 * Date: 16/08/15
 * Time: 12:18
 */

namespace CodeProject\Transformers;
use CodeProject\Entities\User;
use League\Fractal\TransformerAbstract;

class ProjectMemberTransformer extends TransformerAbstract
{
    public function transform(User $member)
    {
	    return [
		  'member_id' => $member->id
	    ];
    }
}