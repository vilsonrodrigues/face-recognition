"""
Need:
python-jose
typing_extensions==4.7.1
"""
import os
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

auth_secret_key = os.getenv("AUTH_SECRET_KEY", default="")
auth_algorithm = os.getenv("AUTH_ALGORITHM", default="HS512")

async def verify_token(token: str = Depends(oauth2_scheme)):
    """
    Verify the validity of a JWT token. If a secret key is not
    set, so the function allow any request.

    Args:
        token: The JWT token obtained from the client.

    Returns:
        None

    Raises:
    - HTTPException: If the token is invalid or cannot be verified, an HTTPException with a
      status code of 401 (Unauthorized) is raised, along with a detail message indicating the
      failure to validate credentials.

    Note:
    This function is intended to be used as a dependency for FastAPI routes to ensure that
    the provided JWT token is valid before allowing access to protected resources.

    Example:
    ``` python
    from fastapi import FastAPI, Depends, HTTPException

    app = FastAPI()

    @app.post("/protected", dependencies=[Depends(verify_token)])
    async def get_protected_data(content: str):
        # Your protected resource logic here
        return {"message": "Access granted to protected data"}
    ```
    """
    # if env is defined
    if auth_secret_key != "":

        credentials_exception = HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
        try:
            jwt.decode(token, auth_secret_key, algorithms=[auth_algorithm])
        except JWTError:
            raise credentials_exception

    # allow any request
    else:
        return
