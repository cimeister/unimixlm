//
//  VZPolicCollectionCell.h
//  Example
//
//  Created by <NAME> on 04.09.14.
//  Copyright (c) 2014 alekoleg. All rights reserved.
//

#import <UIKit/UIKit.h>

extern NSString * const VZPolicCollectionCellIdentifier;

@interface VZPolicCollectionCell : UIView

@property (nonatomic, copy) NSString *reuseIdentifier;

- (id)initWithFrame:(CGRect)frame reuseIdenstifier:(NSString *)identifier;

//layout 
- (void)prepareForReuse;
- (void)setSelected:(BOOL)selected;

@end