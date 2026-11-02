import { Button, Typography } from "antd";
import { MessageSquareQuote } from "lucide-react";
import { Fragment, type ReactNode, useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";

interface SelectableAnswerProps {
  text: string;
  onFollowUp: (quote: string) => void;
}

interface SelectionAction {
  quote: string;
  left: number;
  top: number;
}

const MAX_QUOTE_LENGTH = 360;

function renderInlineMarkdown(value: string, keyPrefix: string): ReactNode[] {
  return value.split(/(\*\*[^*\n]+\*\*|`[^`\n]+`)/g).filter(Boolean).map((part, index) => {
    const key = `${keyPrefix}-${index}`;
    if (part.startsWith("**") && part.endsWith("**")) {
      return <Typography.Text key={key} strong>{part.slice(2, -2)}</Typography.Text>;
    }
    if (part.startsWith("`") && part.endsWith("`")) {
      return <Typography.Text key={key} code>{part.slice(1, -1)}</Typography.Text>;
    }
    return <Fragment key={key}>{part}</Fragment>;
  });
}

function AnswerContent({ text }: { text: string }) {
  return text.split(/\r?\n/).map((rawLine, index) => {
    const line = rawLine.trim();
    const key = `answer-line-${index}`;
    if (!line) return <div key={key} className="answer-line-spacer" aria-hidden="true" />;

    const heading = line.match(/^#{1,6}\s+(.+)$/);
    if (heading) {
      return (
        <Typography.Title key={key} level={5} className="answer-markdown-heading">
          {renderInlineMarkdown(heading[1], key)}
        </Typography.Title>
      );
    }

    const listItem = line.match(/^(\d+[.、]|[-*])\s*(.+)$/);
    if (listItem) {
      return (
        <div key={key} className="answer-markdown-list-item">
          <span className="answer-markdown-list-marker">{listItem[1]}</span>
          <span>{renderInlineMarkdown(listItem[2], key)}</span>
        </div>
      );
    }

    return (
      <Typography.Paragraph key={key} className="answer-markdown-paragraph">
        {renderInlineMarkdown(line, key)}
      </Typography.Paragraph>
    );
  });
}

export function SelectableAnswer({ text, onFollowUp }: SelectableAnswerProps) {
  const contentRef = useRef<HTMLDivElement>(null);
  const [action, setAction] = useState<SelectionAction>();

  useEffect(() => {
    if (!action) return;
    const hide = () => setAction(undefined);
    window.addEventListener("resize", hide);
    window.addEventListener("scroll", hide, true);
    document.addEventListener("pointerdown", hide);
    return () => {
      window.removeEventListener("resize", hide);
      window.removeEventListener("scroll", hide, true);
      document.removeEventListener("pointerdown", hide);
    };
  }, [action]);

  const inspectSelection = () => {
    window.setTimeout(() => {
      const selection = window.getSelection();
      const root = contentRef.current;
      if (!selection || selection.isCollapsed || !selection.rangeCount || !root) {
        setAction(undefined);
        return;
      }
      const range = selection.getRangeAt(0);
      if (!root.contains(range.commonAncestorContainer)) {
        setAction(undefined);
        return;
      }
      const quote = selection.toString().replace(/\s+/g, " ").trim().slice(0, MAX_QUOTE_LENGTH);
      if (quote.length < 2) {
        setAction(undefined);
        return;
      }
      const rect = range.getBoundingClientRect();
      const left = Math.min(window.innerWidth - 54, Math.max(54, rect.left + rect.width / 2));
      const top = Math.max(48, rect.top - 12);
      setAction({ quote, left, top });
    }, 0);
  };

  const followUp = () => {
    if (!action) return;
    onFollowUp(action.quote);
    setAction(undefined);
    window.getSelection()?.removeAllRanges();
  };

  return (
    <>
      <div
        ref={contentRef}
        className="selectable-answer"
        onMouseUp={inspectSelection}
        onTouchEnd={inspectSelection}
        onKeyUp={inspectSelection}
      >
        <AnswerContent text={text} />
      </div>
      {action ? createPortal(
        <Button
          className="selection-follow-up-button"
          size="small"
          icon={<MessageSquareQuote size={14} />}
          style={{ left: action.left, top: action.top }}
          onPointerDown={(event) => event.stopPropagation()}
          onClick={followUp}
        >
          追问
        </Button>,
        document.body,
      ) : null}
    </>
  );
}
