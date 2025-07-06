"use client";

import React, { useState, useEffect } from "react";
import {
  Credentials,
  Theme,
  DocumentFilter,
  ChunkScore,
} from "@/app/types";
import { fetchAllDocumentsMetadata } from "@/app/api"; // Или fetchAllSuggestions

import MEMOInterface from "./MEMOInterface";
import DocumentExplorer from "../Document/DocumentExplorer"; // Убедитесь, что это правильный путь

interface MEMOViewProps {
  selectedTheme: Theme;
  credentials: Credentials;
  addStatusMessage: (
    message: string,
    type: "INFO" | "WARNING" | "SUCCESS" | "ERROR"
  ) => void;
  production: "Local" | "Demo" | "Production";
  currentPage: string;
  documentFilter: DocumentFilter[];
  setDocumentFilter: React.Dispatch<React.SetStateAction<DocumentFilter[]>>;
}

const MEMOView: React.FC<MEMOViewProps> = ({
  credentials,
  selectedTheme,
  addStatusMessage,
  production,
  currentPage,
}) => {
  const [selectedDocument, setSelectedDocument] = useState<string | null>(null);
  const [selectedChunkScore, setSelectedChunkScore] = useState<ChunkScore[]>([]);

  // Новые состояния для списка документов
  const [allDocuments, setAllDocuments] = useState<DocumentFilter[]>([]);
  const [documentFilter, setDocumentFilter] = useState<DocumentFilter[]>([]); // Это будет filteredDocuments

  // Эффект для загрузки всех документов при монтировании
  useEffect(() => {
    const loadAllDocuments = async () => {
      const response = await fetchAllDocumentsMetadata(credentials); // Или fetchAllSuggestions
      if (response && response.documents) { // Или response.suggestions
        setAllDocuments(response.documents);
        setDocumentFilter(response.documents); // Изначально отображаем все документы
      }
    };
    loadAllDocuments();
  }, [credentials]); // Зависимость от credentials

  return (
    <div className="flex flex-row h-full w-full justify-center items-start gap-4 p-4">
      <div className="flex flex-col h-full w-full max-w-[50%]">
        <MEMOInterface
          addStatusMessage={addStatusMessage}
          production={production}
          credentials={credentials}
          selectedTheme={selectedTheme}
          setSelectedDocument={setSelectedDocument}
          setSelectedChunkScore={setSelectedChunkScore}
          currentPage={currentPage}
          documentFilter={documentFilter} // Передаем отфильтрованные документы
          setDocumentFilter={setDocumentFilter} // Позволяем MEMOInterface управлять фильтрацией
          selectedDocument={selectedDocument}
        />
      </div>
      <div className="flex flex-col h-full w-full max-w-[50%]">
        <DocumentExplorer
          selectedTheme={selectedTheme}
          credentials={credentials}
          addStatusMessage={addStatusMessage}
          selectedDocument={selectedDocument}
          setSelectedDocument={setSelectedDocument}
          documentFilter={documentFilter} // Передаем список документов
          setDocumentFilter={setDocumentFilter} // Передаем функцию для обновления фильтров
          production={production}
        />
      </div>
    </div>
  );
};

export default MEMOView;
