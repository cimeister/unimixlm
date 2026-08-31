import React from 'react';
import DivWrapper from './';
import { mount } from 'enzyme';

describe('<DivWrapper />', () => {
  it('renders smoke test', () => {
    const comp = mount(<DivWrapper />);
    const ErrorBoundary = comp.find('ErrorBoundary');
    expect(ErrorBoundary.length).toBe(1);
  });
  // describe('renders passed props', () => {
  //   const component = mount
  // })
});