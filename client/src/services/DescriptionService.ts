import http from "../http-common";

const getMore = (description: string): Promise<any> => {
    let data = {
        "description": description
    }

    return http.post("/additional_results", data = data, {
        headers: {
            "Content-Type": "application/json",
        }
    });
};

const FileUploadService = {
    getMore: getMore
};

export default FileUploadService;
