import http from "../http-common";

const upload = (file: File, onUploadProgress: any): Promise<any> => {
    let formData = new FormData();

    formData.append("file", file);

    return http.post("/upload", formData, {
        headers: {
            "Content-Type": "text/csv",
        },
        onUploadProgress,
    });
};

const FileService = {
    upload: upload
};

export default FileService;


