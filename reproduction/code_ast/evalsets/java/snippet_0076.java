package com.mouse.api.service;

import com.mouse.dao.entity.resource.GoodsSpecificationEntity;

import java.util.List;
import java.util.Optional;

/**
 * @author ; lidongdong
 * @Description
 * @Date 2019-12-21
 */
public interface GoodsSpecificationService {
    /**
     * 根据商品ID查询商品规格
     *
     * @param goodsId 商品ID
     * @return
     */
    Optional<List<GoodsSpecificationEntity>> findByGoodsId(Integer goodsId);
}