"""Anonymous workspaces.

There are no accounts. The first upload creates a workspace and hands the browser
an HMAC-signed cookie holding its id; every later request is scoped to it. A
client can't guess or forge another workspace's cookie without the server key.
"""

import base64
import hashlib
import hmac
import uuid
from typing import Annotated

from fastapi import Depends, Request, Response

from prism.db.models import Workspace

from .deps import SessionDep, SettingsDep

COOKIE_NAME = "prism_ws"
COOKIE_MAX_AGE = 365 * 24 * 3600


def _mac(workspace_id: uuid.UUID, secret: str) -> str:
    digest = hmac.new(secret.encode(), b"workspace:" + workspace_id.bytes, hashlib.sha256)
    return base64.urlsafe_b64encode(digest.digest()).rstrip(b"=").decode()


def sign(workspace_id: uuid.UUID, secret: str) -> str:
    return f"{workspace_id.hex}.{_mac(workspace_id, secret)}"


def verify(token: str | None, secret: str) -> uuid.UUID | None:
    if not token or token.count(".") != 1:
        return None
    raw_id, mac = token.split(".")
    try:
        workspace_id = uuid.UUID(hex=raw_id)
    except ValueError:
        return None
    if not hmac.compare_digest(mac, _mac(workspace_id, secret)):
        return None
    return workspace_id


def current_workspace(request: Request, settings: SettingsDep) -> uuid.UUID | None:
    """The caller's workspace id, or None if they have no valid cookie yet."""
    return verify(request.cookies.get(COOKIE_NAME), settings.secret_key.get_secret_value())


async def ensure_workspace(
    response: Response,
    workspace_id: Annotated[uuid.UUID | None, Depends(current_workspace)],
    session: SessionDep,
    settings: SettingsDep,
) -> uuid.UUID:
    """Like current_workspace, but creates one (and sets the cookie) when missing."""
    if workspace_id is not None and await session.get(Workspace, workspace_id) is not None:
        return workspace_id

    workspace = Workspace()
    session.add(workspace)
    await session.flush()
    response.set_cookie(
        COOKIE_NAME,
        sign(workspace.id, settings.secret_key.get_secret_value()),
        max_age=COOKIE_MAX_AGE,
        httponly=True,
        secure=settings.cookie_secure,
        samesite="lax",
    )
    return workspace.id


CurrentWorkspace = Annotated[uuid.UUID | None, Depends(current_workspace)]
EnsuredWorkspace = Annotated[uuid.UUID, Depends(ensure_workspace)]
