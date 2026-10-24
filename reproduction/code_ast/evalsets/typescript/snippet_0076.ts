import { Dataset } from "chipster-js-common";
import { Tags, TypeTagService } from "../../shared/services/typetag.service";
import { SelectionOption } from "../../views/sessions/session/SelectionOption";
import { SessionData } from "../session/session-data";

export default class TSV2File {
  public dataset: Dataset;

  private headers: Array<string>;
  private headersWithIdentifierFix: Array<string>;
  private headersHardCodedByType: Array<string>;

  private body: Array<Array<string>>;

  private isMissingHeaderColumn = false;
  private isFirstDataRowHeader = false;
  private hasHeadersHardcodedByType = false;

  /**
   * Replace "" -> <empty>, " " -> <space>, multiple " " -> <spaces>
   */
  private static spreadSheetReplace(headers: Array<string>): Array<string> {
    return headers.map((header) => {
      if (header === "") {
        return "&lt;empty&gt;";
      }

      if (header === " ") {
        return "&lt;space&gt;";
      }
      if (/^\s$/.test(header)) {
        return "&lt;spaces&gt;";
      }

      return header;
    });
  }

  private static parameterReplace(header): string {
    if (header === "") {
      return "untitled";
    }
    if (header === " ") {
      return "<space>";
    }
    if (/^\s$/.test(header)) {
      return "<spaces>";
    }
    return header;
  }

  private static getIsMissingHeaderColumn(tsvArray: Array<Array<string>>): boolean {
    if (tsvArray.length <= 2) {
      // have to guess
      return false;
    }
    return tsvArray[0].length < tsvArray[1].length;
  }

  public static create(
    tsvArray: Array<Array<string>>,
    dataset: Dataset,
    sessionData: SessionData,
    typeTagService: TypeTagService
  ): TSV2File {
    // all these neded when construction new TSV2File
    let headersWithIdentifierFix: Array<string>;
    let headersHardcodedByType: Array<string>;
    let hasHeadersHardcodedByType = false;
    let isFirstDataRowHeader = false;
    let isMissingHeaderColumn = false;
    let body: Array<Array<string>>;
    let headers: Array<string>;

    // type-service gives the column headers for some file types
    const headersFromTypeString = typeTagService.get(sessionData, dataset, Tags.COLUMN_TITLES);

    // headers hard coded by type
    if (headersFromTypeString) {
      // create a new first row from the column titles
      // FIXME check sizes
      hasHeadersHardcodedByType = true;
      headersHardcodedByType = headersFromTypeString.split("\t");
      body = tsvArray;
    } else {
      // whether the first data row is header row or not

      if (typeTagService.has(sessionData, dataset, Tags.NO_TITLE_ROW)) {
        // no headers in data, create new header row with appropriate number of empty columns

        if (tsvArray.length > 0) {
          headers = new Array<string>(tsvArray[0].length);
        } else {
          headers = [];
        }
        body = tsvArray;
      } else {
        // first data row is header row
        isFirstDataRowHeader = true;

        if (tsvArray.length > 0) {
          headers = tsvArray[0];
          body = tsvArray.slice(1); // will return [] if index out of bounds
          isMissingHeaderColumn = this.getIsMissingHeaderColumn(tsvArray);
          if (isMissingHeaderColumn) {
            headersWithIdentifierFix = [""].concat(headers);
          }
        } else {
          headers = [];
          body = [];
        }
      }
    }
    return new TSV2File(
      dataset,
      headers,
      headersWithIdentifierFix,
      headersHardcodedByType,
      hasHeadersHardcodedByType,
      body,
      isMissingHeaderColumn,
      isFirstDataRowHeader
    );
  }

  private constructor(
    dataset: Dataset,
    headers: Array<string>,
    headersWithIdentifierFix: Array<string>,
    headersHardcodedByType: Array<string>,
    hasHeadersHardcodedByType: boolean,
    body: Array<Array<string>>,
    isMissingHeaderColumn: boolean,
    isFirstDataRowHeader: boolean
  ) {
    this.dataset = dataset;
    this.headers = headers;
    this.headersWithIdentifierFix = headersWithIdentifierFix;
    this.headersHardCodedByType = headersHardcodedByType;
    this.hasHeadersHardcodedByType = hasHeadersHardcodedByType;
    this.body = body;
    this.isMissingHeaderColumn = isMissingHeaderColumn;
    this.isFirstDataRowHeader = isFirstDataRowHeader;
  }

  public getBody(): Array<Array<string>> {
    return this.body;
  }

  public getHeadersForSpreadSheet(): Array<string> {
    if (this.hasHeadersHardcodedByType) {
      return this.headersHardCodedByType;
    }
    if (this.isFirstDataRowHeader) {
      return this.isMissingHeaderColumn
        ? [this.headersWithIdentifierFix[0]].concat(TSV2File.spreadSheetReplace(this.headersWithIdentifierFix.slice(1)))
        : TSV2File.spreadSheetReplace(this.headers);
    }
    return this.headers;
  }

  public getHeadersForParameter(): Array<SelectionOption> {
    let sourceHeaders;

    // order matters here
    if (this.hasHeadersHardcodedByType || !this.isFirstDataRowHeader) {
      if (this.getBody().length > 0) {
        sourceHeaders = this.getBody()[0];
      } else {
        sourceHeaders = [];
      }
    } else if (this.isMissingHeaderColumn) {
      sourceHeaders = this.headersWithIdentifierFix;
    } else {
      sourceHeaders = this.headers;
    }
    return sourceHeaders.map((header) => ({
      id: header,
      displayName: TSV2File.parameterReplace(header),
    }));
  }
}