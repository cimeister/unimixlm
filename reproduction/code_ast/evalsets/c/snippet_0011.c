/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   system_tree_new.c                                  :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: sid <<EMAIL>>               +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2016/03/01 14:10:06 by sid               #+#    #+#             */
/*   Updated: 2017/01/24 06:47:03 by angagnie         ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "system_tree.h"
#include "ggl.h"

/*
** -----===== Constructors =====-----
*/

t_gnode		*gnode_alloc(size_t type_size)
{
	t_gnode		*ans;

	if ((ans = (t_gnode *)malloc(type_size)))
	{
		ftl_init((t_list *)ans, type_size);
		ans->parent = NULL;
		ans->pos = NEW_VEC3(0, 0, 0);
		ans->alpha = 0;
		ans->beta = 0;
		ans->gamma = 0;
		ans->scale = 1;
		ans->is_obj = (type_size == sizeof(t_obj));
	}
	return (ans);
}

t_obj		*obj_alloc(char *str)
{
	t_obj	*ans;

	if ((ans = (t_obj *)gnode_alloc(sizeof(t_obj))))
	{
		ans->vertices = NEW_ARRAY(t_vrtx);
		ans->faces = NEW_ARRAY(t_face);
		if (!(ans->name = ft_strdup(str)))
			ft_putstr_fd("ERROR\tobj_alloc : Malloc failed\n", 2);
		ans->dim = 0;
	}
	return (ans);
}

t_camera	*camera_alloc(void)
{
	t_camera *ans;

	if ((ans = (t_camera *)gnode_alloc(sizeof(t_camera))))
	{
		ans->fov = 60;
		ans->ratio = (t_pnt2i){16, 9};
	}
	return (ans);
}

t_obj		obj_new(char *str)
{
	t_obj	ans;

	ftl_init((t_list *)&ans.node, sizeof(t_obj));
	ans.node.parent = NULL;
	ans.node.pos = NEW_VEC3(0, 0, 0);
	ans.node.alpha = 0;
	ans.node.beta = 0;
	ans.node.gamma = 0;
	ans.node.scale = 1;
	ans.vertices = NEW_ARRAY(t_vrtx);
	ans.faces = NEW_ARRAY(t_vrtx);
	if (!(ans.name = ft_strdup(str)))
		ft_putstr_fd("ERROR\tobj_new : Malloc failed\n", 2);
	ans.dim = 0;
	ans.node.is_obj = 1;
	return (ans);
}