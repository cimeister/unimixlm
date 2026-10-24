/*
 * Copyright (c) 2022 by <NAME> <https://miloszgilga.pl>
 *
 * File name: CmsTextareasAddNewArea.tsx
 * Last modified: 05/04/2022, 21:17
 * Project name: digititles-imagine
 *
 * Licensed under the MIT license; you may not use this file except in compliance with the License.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
 * documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to the following conditions:
 *
 * THE ABOVE COPYRIGHT NOTICE AND THIS PERMISSION NOTICE SHALL BE INCLUDED IN ALL
 * COPIES OR SUBSTANTIAL PORTIONS OF THE SOFTWARE.
 */

import * as React from 'react';
import { useDispatch, useSelector } from 'react-redux';

import { RootState } from '../../../redux/store';
import { SoftwareModel } from '../../../redux/redux-models/ProjectModel';
import { InitStateAPItypes } from '../../../redux/redux-api-thunk/initialState';
import { ReduxProjFormActions } from '../../../redux/redux-subreducers/redux-project-form/actions';
import { DiscretteProjectSections } from '../../../redux/redux-subreducers/redux-project-form/types';

import { CmsTextareasAddNewAreaButton } from '../CmsTextareasListsStructure.styles';


interface PropsProvider {
    section: DiscretteProjectSections;
    addValuePayload?: SoftwareModel;
}

const CmsTextareasAddNewArea: React.FC<PropsProvider> = ({ section, addValuePayload }): JSX.Element => {

    const { projectDataForm }: InitStateAPItypes = useSelector((state: RootState) => state.reduxGlobalReducer);
    const dispatcher = useDispatch();

    const buttonTitle = projectDataForm[section].length < 7
        ? `Click to add new section in ${section.toLocaleLowerCase()}`
        : 'You cannot add new section';

    const handleAddNewSection = (): void => {
        if (projectDataForm[section].length < 7) {
            dispatcher(ReduxProjFormActions.addProjectArrayElement(section, addValuePayload || ''));
        }
    };

    return (
        <CmsTextareasAddNewAreaButton
            title = {buttonTitle}
            type = 'button'
            onClick = {handleAddNewSection}
            disabled = {projectDataForm[section].length >= 7}
        >
            Add new section ({projectDataForm[section].length} of 7)
        </CmsTextareasAddNewAreaButton>
    );
};

export default CmsTextareasAddNewArea;