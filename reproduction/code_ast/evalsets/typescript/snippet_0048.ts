import { useCallback } from "react";
import { useHistory } from "react-router-dom";
import { gql, useQuery } from "@apollo/client";
import { formatError } from "util/error";
import * as models from "models";

type TData = {
  entity: models.Entity;
};

const DATE_CREATED_FIELD = "createdAt";

export type Props = {
  entityId: string;
};

const useEntityFieldLinkList = ({ entityId }: Props) => {
  const { data, error } = useQuery<TData>(GET_FIELDS, {
    variables: {
      id: entityId,
      orderBy: {
        [DATE_CREATED_FIELD]: models.SortOrder.Asc,
      },
    },
  });
  const history = useHistory();
  const handleFieldAdd = useCallback(
    (field: models.EntityField) => {
      const fieldUrl = `/${data?.entity.appId}/entities/${entityId}/fields/${field.id}`;
      history.push(fieldUrl);
    },
    [data, history, entityId]
  );

  const errorMessage = formatError(error);

  const result = {
    data,
    error,
    errorMessage,
    handleFieldAdd
  };

  return result;
};

export default useEntityFieldLinkList;

/**@todo: expand search on other field  */
export const GET_FIELDS = gql`
  query getEntityFields(
    $id: String!
    $orderBy: EntityFieldOrderByInput
    $whereName: StringFilter
  ) {
    entity(where: { id: $id }) {
      id
      appId
      fields(where: { displayName: $whereName }, orderBy: $orderBy) {
        id
        displayName
        name
        dataType
        required
        searchable
        description
        properties
      }
    }
  }
`;