"""gRPC server for converter daemon artifact transport."""

from __future__ import annotations

import argparse
import os
from collections.abc import Iterable, Iterator
from concurrent import futures
from typing import TYPE_CHECKING, Any, Protocol, cast

try:
    import grpc
except ModuleNotFoundError:  # pragma: no cover
    grpc = None

from converter_grpc import converter_pb2 as _converter_pb2
from onnx_converter.converter.core import ConversionRequest, convert_artifact_bytes
from onnx_converter.errors import ConversionError

converter_pb2: Any = cast(Any, _converter_pb2)
if grpc is not None:
    from converter_grpc import converter_pb2_grpc as _converter_pb2_grpc

    converter_pb2_grpc: Any = cast(Any, _converter_pb2_grpc)
else:  # pragma: no cover
    converter_pb2_grpc = None

if TYPE_CHECKING:
    from grpc import ServicerContext
else:
    ServicerContext = Any


class _GrpcStatusCode(Protocol):
    """Subset of grpc.StatusCode enum used by this module."""

    INVALID_ARGUMENT: object
    INTERNAL: object


class _GrpcServer(Protocol):
    """Subset of grpc.Server API used by this module."""

    def add_insecure_port(self, address: str) -> int:
        """Bind server to an insecure address."""
        ...

    def start(self) -> None:
        """Start serving requests."""
        ...

    def wait_for_termination(self) -> None:
        """Block until server termination."""
        ...


class _GrpcRuntime(Protocol):
    """Subset of grpc module API used by this module."""

    StatusCode: _GrpcStatusCode

    def server(self, executor: futures.Executor) -> _GrpcServer:
        """Build grpc server instance."""
        ...


class _GrpcStubs(Protocol):
    """Subset of generated stub helpers used by this module."""

    def add_ConverterServiceServicer_to_server(
        self,
        servicer: object,
        server: _GrpcServer,
    ) -> None:
        """Register converter service implementation."""
        ...


def _require_grpc_runtime() -> _GrpcRuntime:
    """Return grpc module or raise an actionable runtime error."""
    if grpc is None:
        raise RuntimeError(
            "grpcio is required to run converter-grpc. Install with extra: .[grpc]"
        )
    return cast(_GrpcRuntime, grpc)


def _require_grpc_stubs() -> _GrpcStubs:
    """Return generated grpc stubs module or raise an actionable runtime error."""
    if converter_pb2_grpc is None:
        raise RuntimeError(
            "gRPC stubs are unavailable. Install grpc dependencies with extra: .[grpc]"
        )
    return cast(_GrpcStubs, converter_pb2_grpc)


def _iter_chunks(payload: bytes, *, chunk_size: int = 1 << 20) -> Iterator[bytes]:
    """Yield payload bytes in fixed-size chunks."""
    offset = 0
    total = len(payload)
    while offset < total:
        end = min(offset + chunk_size, total)
        yield payload[offset:end]
        offset = end


class ConverterGrpcService:
    """gRPC converter service implementation."""

    def Convert(
        self,
        request_iterator: Iterable[object],
        context: ServicerContext,
    ) -> Iterator[object]:  # noqa: N802
        """Receive artifact chunks, run conversion, and stream ONNX output chunks."""
        grpc_runtime = _require_grpc_runtime()
        metadata = None
        chunks: list[bytes] = []
        for request in request_iterator:
            chunk = cast(Any, request)
            payload_kind = chunk.WhichOneof("payload")
            if payload_kind == "metadata":
                if metadata is not None:
                    context.abort(
                        grpc_runtime.StatusCode.INVALID_ARGUMENT,
                        "metadata provided more than once",
                    )
                metadata = chunk.metadata
                continue
            if payload_kind != "data":
                continue
            data = bytes(chunk.data)
            if data:
                chunks.append(data)

        if metadata is None:
            context.abort(
                grpc_runtime.StatusCode.INVALID_ARGUMENT,
                "missing conversion metadata",
            )
        if not chunks:
            context.abort(
                grpc_runtime.StatusCode.INVALID_ARGUMENT,
                "missing artifact payload",
            )

        md = cast(Any, metadata)
        framework = str(md.framework).strip().lower()
        input_shape = (
            tuple(int(v) for v in md.input_shape)
            if getattr(md, "input_shape", None)
            else None
        )
        n_features_raw = int(md.n_features) if int(md.n_features) > 0 else None
        request = ConversionRequest(
            framework=framework,  # type: ignore[arg-type]
            filename=str(md.filename or "artifact.bin"),
            expected_sha256=(str(md.expected_sha256).strip() or None),
            input_shape=input_shape,
            n_features=n_features_raw,
            opset_version=int(md.opset_version or 14),
            # gRPC transport does not allow clients to opt into unsafe deserialization.
            allow_unsafe=False,
        )
        try:
            input_sha, outcome = convert_artifact_bytes(b"".join(chunks), request)
        except ValueError as exc:
            context.abort(grpc_runtime.StatusCode.INVALID_ARGUMENT, str(exc))
        except ConversionError as exc:
            context.abort(grpc_runtime.StatusCode.INVALID_ARGUMENT, str(exc))
        except Exception as exc:
            context.abort(grpc_runtime.StatusCode.INTERNAL, str(exc))

        yield converter_pb2.ConvertReplyChunk(
            result=converter_pb2.ConvertResult(
                input_sha256=input_sha,
                output_sha256=outcome.output_sha256,
                output_filename=outcome.output_filename,
                output_size_bytes=outcome.output_size_bytes,
            )
        )
        for chunk in _iter_chunks(outcome.output_bytes):
            yield converter_pb2.ConvertReplyChunk(data=chunk)


def create_server(*, host: str, port: int, max_workers: int = 8) -> _GrpcServer:
    """Create gRPC server instance."""
    grpc_runtime = _require_grpc_runtime()
    stubs = _require_grpc_stubs()
    server = grpc_runtime.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    stubs.add_ConverterServiceServicer_to_server(
        ConverterGrpcService(),
        server,
    )
    server.add_insecure_port(f"{host}:{port}")
    return server


def main() -> None:
    """Run converter daemon gRPC entrypoint."""
    parser = argparse.ArgumentParser(description="ONNX converter daemon gRPC server.")
    parser.add_argument(
        "--host",
        default=os.getenv("CONVERTER_GRPC_HOST", "0.0.0.0"),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("CONVERTER_GRPC_PORT", "8091")),
    )
    parser.add_argument("--max-workers", type=int, default=8)
    args = parser.parse_args()
    server = create_server(host=args.host, port=args.port, max_workers=args.max_workers)
    server.start()
    print(f"converter grpc listening on {args.host}:{args.port}")
    server.wait_for_termination()


if __name__ == "__main__":
    main()
