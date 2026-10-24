// UserOptions -----------------------------------------------------------------

// List Users that match search criteria, offering callbacks for adding,
// editing, and removing Users.

// External Modules ----------------------------------------------------------

import React, {useContext, useEffect, useState} from "react";
import Button from "react-bootstrap/Button";
import Container from "react-bootstrap/Container";
import Col from "react-bootstrap/Col";
import Row from "react-bootstrap/Row";
import Table from "react-bootstrap/Table";

// Internal Modules ----------------------------------------------------------

import CheckBox from "../general/CheckBox";
import LoadingProgress from "../general/LoadingProgress";
import Pagination from "../general/Pagination";
import SearchBar from "../general/SearchBar";
import LoginContext from "../login/LoginContext";
import {HandleAction, HandleBoolean, HandleUser, HandleValue, Scope} from "../../types";
import useFetchUsers from "../../hooks/useFetchUsers";
import User from "../../models/User";
import logger from "../../util/ClientLogger";
import {listValue} from "../../util/Transformations";

// Incoming Properties -------------------------------------------------------

export interface Props {
    handleAdd?: HandleAction;           // Handle request to add a User [not allowed]
    handleEdit?: HandleUser;            // Handle request to edit a User [not allowed]
}

// Component Details ---------------------------------------------------------

const UserOptions = (props: Props) => {

    const loginContext = useContext(LoginContext);

    const [active, setActive] = useState<boolean>(false);
    const [availables, setAvailables] = useState<User[]>([]);
    const [currentPage, setCurrentPage] = useState<number>(1);
    const [pageSize] = useState<number>(100);
    const [searchText, setSearchText] = useState<string>("");

    const fetchUsers = useFetchUsers({
        active: active,
        alertPopup: false,
        currentPage: currentPage,
        pageSize: pageSize,
        username: (searchText.length > 0) ? searchText : undefined,
    });

    useEffect(() => {

        logger.debug({
            context: "UserOptions.useEffect",
            active: active,
            searchText: searchText,
        });

        const isSuperuser = loginContext.validateScope(Scope.SUPERUSER);
        if (isSuperuser) {
            setAvailables(fetchUsers.users);
        } else {
            setAvailables([]);
        }

    }, [loginContext, loginContext.data.loggedIn,
        active, searchText,
        fetchUsers.users]);

    const handleActive: HandleBoolean = (theActive) => {
        setActive(theActive);
    }

    const handleAdd: HandleAction = () => {
        if (props.handleAdd) {
            props.handleAdd();
        }
    }

    const handleChange: HandleValue = (theSearchText) => {
        setSearchText(theSearchText);
    }

    const handleEdit: HandleUser = (theUser) => {
        if (props.handleEdit) {
            props.handleEdit(theUser);
        }
    }

    const handleNext: HandleAction = () => {
        setCurrentPage(currentPage + 1);
    }

    const handlePrevious: HandleAction = () => {
        setCurrentPage(currentPage - 1);
    }

    return (
        <Container fluid id="UserOptions">

            <LoadingProgress
                error={fetchUsers.error}
                loading={fetchUsers.loading}
                title="Selected Users"
            />

            <Row className="mb-3">
                <Col className="col-6">
                    <SearchBar
                        autoFocus
                        handleChange={handleChange}
                        htmlSize={50}
                        label="Search For Users:"
                        placeholder="Search by all or part of username"
                    />
                </Col>
                <Col>
                    <CheckBox
                        handleChange={handleActive}
                        label="Active Users Only?"
                        name="activeOnly"
                        value={active}
                    />
                </Col>
                <Col className="text-end">
                    <Pagination
                        currentPage={currentPage}
                        handleNext={handleNext}
                        handlePrevious={handlePrevious}
                        lastPage={(availables.length === 0) ||
                            (availables.length < pageSize)}
                        variant="secondary"
                    />
                </Col>
                <Col className="text-end">
                    <Button
                        disabled={!props.handleAdd}
                        onClick={handleAdd}
                        size="sm"
                        variant="primary"
                    >Add</Button>
                </Col>
            </Row>

            <Row className="g-2">
                <Table
                    bordered={true}
                    hover={true}
                    size="sm"
                    striped={true}
                >

                    <thead>
                    <tr className="table-secondary">
                        <th scope="col">Username</th>
                        <th scope="col">Active</th>
                        <th scope="col">Name</th>
                        <th scope="col">Scope</th>
                    </tr>
                    </thead>

                    <tbody>
                    {availables.map((user, rowIndex) => (
                        <tr
                            className="table-default"
                            key={1000 + (rowIndex * 100)}
                            onClick={props.handleEdit ? (() => handleEdit(user)) : undefined}
                        >
                            <td key={1000 + (rowIndex * 100) + 1}>
                                {user.username}
                            </td>
                            <td key={1000 + (rowIndex * 100) + 2}>
                                {listValue(user.active)}
                            </td>
                            <td key={1000 + (rowIndex * 100) + 3}>
                                {user.name}
                            </td>
                            <td key={1000 + (rowIndex * 100) + 4}>
                                {user.scope}
                            </td>
                        </tr>
                    ))}
                    </tbody>

                </Table>
            </Row>

            <Row className="mb-3">
                <Col className="text-end">
                    <Button
                        disabled={!props.handleAdd}
                        onClick={handleAdd}
                        size="sm"
                        variant="primary"
                    >Add</Button>
                </Col>
            </Row>

        </Container>

    )

}

export default UserOptions;